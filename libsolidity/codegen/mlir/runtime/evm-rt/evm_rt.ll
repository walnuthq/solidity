; evm-rt v0: EVM-semantics 256-bit runtime, written directly as LLVM IR.
;
; Rationale: the operations need i256/i512 arithmetic with EVM-exact edge
; semantics; LLVM's backend legalization (including ExpandLargeDivRem for
; the division family) lowers these for any target - rv32im for zkVM
; guests, the host for JIT differentials - with no C-frontend _BitInt
; width limits in the way. Optimized Knuth-D limbs and zkVM accelerator
; routing replace these on the hot path later.
;
; ABI (matches EVMToLLVM's lowering): result pointer first, then operand
; pointers; every pointer refers to a 32-byte little-endian-limb word.
;
; SPDX-License-Identifier: GPL-3.0

; ---- unsigned division: x / 0 == 0 ----
define void @__evm_rt_div(ptr %r, ptr %a, ptr %b) {
entry:
  %av = load i256, ptr %a
  %bv = load i256, ptr %b
  %z = icmp eq i256 %bv, 0
  br i1 %z, label %retzero, label %dodiv
retzero:
  store i256 0, ptr %r
  ret void
dodiv:
  %q = udiv i256 %av, %bv
  store i256 %q, ptr %r
  ret void
}

; ---- signed division: x / 0 == 0, MIN / -1 == MIN ----
define void @__evm_rt_sdiv(ptr %r, ptr %a, ptr %b) {
entry:
  %av = load i256, ptr %a
  %bv = load i256, ptr %b
  %z = icmp eq i256 %bv, 0
  br i1 %z, label %retzero, label %chkmin
retzero:
  store i256 0, ptr %r
  ret void
chkmin:
  %ismin = icmp eq i256 %av, u0x8000000000000000000000000000000000000000000000000000000000000000
  %isneg1 = icmp eq i256 %bv, -1
  %ovf = and i1 %ismin, %isneg1
  br i1 %ovf, label %retmin, label %dodiv
retmin:
  store i256 %av, ptr %r
  ret void
dodiv:
  %q = sdiv i256 %av, %bv
  store i256 %q, ptr %r
  ret void
}

; ---- unsigned modulo: x mod 0 == 0 ----
define void @__evm_rt_mod(ptr %r, ptr %a, ptr %b) {
entry:
  %av = load i256, ptr %a
  %bv = load i256, ptr %b
  %z = icmp eq i256 %bv, 0
  br i1 %z, label %retzero, label %dorem
retzero:
  store i256 0, ptr %r
  ret void
dorem:
  %m = urem i256 %av, %bv
  store i256 %m, ptr %r
  ret void
}

; ---- signed modulo: x smod 0 == 0, MIN smod -1 == 0, sign of dividend ----
define void @__evm_rt_smod(ptr %r, ptr %a, ptr %b) {
entry:
  %av = load i256, ptr %a
  %bv = load i256, ptr %b
  %z = icmp eq i256 %bv, 0
  br i1 %z, label %retzero, label %chkmin
chkmin:
  %ismin = icmp eq i256 %av, u0x8000000000000000000000000000000000000000000000000000000000000000
  %isneg1 = icmp eq i256 %bv, -1
  %ovf = and i1 %ismin, %isneg1
  br i1 %ovf, label %retzero, label %dorem
retzero:
  store i256 0, ptr %r
  ret void
dorem:
  %m = srem i256 %av, %bv
  store i256 %m, ptr %r
  ret void
}

; ---- addmod: (a + b) mod c over a 512-bit intermediate; c == 0 -> 0 ----
define void @__evm_rt_addmod(ptr %r, ptr %a, ptr %b, ptr %c) {
entry:
  %cv = load i256, ptr %c
  %z = icmp eq i256 %cv, 0
  br i1 %z, label %retzero, label %compute
retzero:
  store i256 0, ptr %r
  ret void
compute:
  %av = load i256, ptr %a
  %bv = load i256, ptr %b
  %aw = zext i256 %av to i512
  %bw = zext i256 %bv to i512
  %cw = zext i256 %cv to i512
  %sum = add i512 %aw, %bw
  %m = urem i512 %sum, %cw
  %t = trunc i512 %m to i256
  store i256 %t, ptr %r
  ret void
}

; ---- mulmod: (a * b) mod c over a 512-bit intermediate; c == 0 -> 0 ----
define void @__evm_rt_mulmod(ptr %r, ptr %a, ptr %b, ptr %c) {
entry:
  %cv = load i256, ptr %c
  %z = icmp eq i256 %cv, 0
  br i1 %z, label %retzero, label %compute
retzero:
  store i256 0, ptr %r
  ret void
compute:
  %av = load i256, ptr %a
  %bv = load i256, ptr %b
  %aw = zext i256 %av to i512
  %bw = zext i256 %bv to i512
  %cw = zext i256 %cv to i512
  %prod = mul i512 %aw, %bw
  %m = urem i512 %prod, %cw
  %t = trunc i512 %m to i256
  store i256 %t, ptr %r
  ret void
}

; ---- exp: wrapping square-and-multiply ----
define void @__evm_rt_exp(ptr %r, ptr %basep, ptr %expp) {
entry:
  %b0 = load i256, ptr %basep
  %e0 = load i256, ptr %expp
  br label %loop
loop:
  %res = phi i256 [ 1, %entry ], [ %res2, %body ]
  %b = phi i256 [ %b0, %entry ], [ %b2, %body ]
  %e = phi i256 [ %e0, %entry ], [ %e2, %body ]
  %done = icmp eq i256 %e, 0
  br i1 %done, label %exit, label %body
body:
  %lsb = trunc i256 %e to i1
  %mul = mul i256 %res, %b
  %res2 = select i1 %lsb, i256 %mul, i256 %res
  %b2 = mul i256 %b, %b
  %e2 = lshr i256 %e, 1
  br label %loop
exit:
  store i256 %res, ptr %r
  ret void
}

; ---- byte: big-endian byte extraction; index >= 32 -> 0 ----
define void @__evm_rt_byte(ptr %r, ptr %ip, ptr %xp) {
entry:
  %i = load i256, ptr %ip
  %big = icmp uge i256 %i, 32
  br i1 %big, label %retzero, label %pick
retzero:
  store i256 0, ptr %r
  ret void
pick:
  %x = load i256, ptr %xp
  %d = sub i256 31, %i
  %sh = mul i256 %d, 8
  %v = lshr i256 %x, %sh
  %byte = and i256 %v, 255
  store i256 %byte, ptr %r
  ret void
}

; ---- signextend: extend from byte index b; b >= 31 -> x ----
define void @__evm_rt_signextend(ptr %r, ptr %bp, ptr %xp) {
entry:
  %b = load i256, ptr %bp
  %x = load i256, ptr %xp
  %big = icmp uge i256 %b, 31
  br i1 %big, label %asis, label %ext
asis:
  store i256 %x, ptr %r
  ret void
ext:
  ; shift the sign bit of byte b (bit 8b+7) up to bit 255, then back down
  %m8 = mul i256 %b, 8
  %n = sub i256 248, %m8
  %sl = shl i256 %x, %n
  %sr = ashr i256 %sl, %n
  store i256 %sr, ptr %r
  ret void
}
