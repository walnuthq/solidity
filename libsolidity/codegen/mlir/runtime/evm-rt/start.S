# Minimal freestanding entry point for linked RV32 test artifacts.
# The Linux-user harness (qemu-riscv32, CI) replaces this with a real
# test driver; locally it proves the full link.
    .globl _start
_start:
    li a7, 93        # exit
    li a0, 0
    ecall
