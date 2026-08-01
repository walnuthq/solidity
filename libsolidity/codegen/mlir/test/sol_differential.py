#!/usr/bin/env python3
"""
Differential for the sol rung: Solidity through the whole MLIR ladder against
the same source through solc, both runtime objects installed at an address
and every ABI selector called on each.

Only possible since SolToYul started emitting a dispatcher - before that the
ladder's output had no entry point to call. It earned its keep immediately,
catching a dispatcher that read past the end of calldata instead of reverting.

  anvil --silent --port 8546 &
  python3 sol_differential.py <dir-of-sol-files> <max-files>
"""
import subprocess, sys, re, pathlib
sys.path.insert(0, 'libsolidity/codegen/mlir/test')
import deploy_differential as D
SOLC='./build/solc/solc'; S2E='./build/libsolidity/codegen/mlir/tools/sol2evm'
def rpc_reachable(url):
    """CTest convention: 77 means skip. A missing node is not a failure."""
    import urllib.request, json as _json
    try:
        request = urllib.request.Request(
            url,
            data=_json.dumps({"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(request, timeout=5).read()
        return True
    except Exception:
        return False

if not rpc_reachable('http://127.0.0.1:8546'):
    print('no JSON-RPC node at http://127.0.0.1:8546 - skipping'); sys.exit(77)

rpc = D.Rpc('http://127.0.0.1:8546')
addr_n = 0x700000
ok = bad = skip = 0
for src in sorted(pathlib.Path(sys.argv[1]).rglob('*.sol'))[:int(sys.argv[2])]:
    r = subprocess.run([S2E, str(src), '--hex'], capture_output=True, text=True, timeout=60)
    mine = {}
    for line in r.stdout.splitlines():
        m = re.match(r'HEX contract="([^"]+)" ([0-9a-f]+)', line)
        if m: mine[m.group(1).split(':')[-1]] = m.group(2)
    if not mine: continue
    ref = subprocess.run([SOLC,'--via-ir','--optimize','--bin','--hashes',str(src)],
                         capture_output=True, text=True)
    # per-contract runtime code + selectors
    cur=None; want=False; refs={}
    for line in ref.stdout.splitlines():
        s=line.strip()
        if s.startswith('======='): cur=s.strip('= ').split(':')[-1]; refs.setdefault(cur,['',[]]); want=False
        elif cur is None: continue
        elif s == 'Binary:': want=True
        elif want:
            if s and all(c in '0123456789abcdef' for c in s): refs[cur][0]=s
            want=False
        else:
            m=re.match(r'^([0-9a-f]{8}):', s)
            if m: refs[cur][1].append(m.group(1))
    for name, code in mine.items():
        if name not in refs or not refs[name][0] or '__$' in refs[name][0]: skip+=1; continue
        rcode, sels = refs[name]
        if not sels: skip+=1; continue
        # Same snapshot for each so both land at the same address and neither
        # sees the other's storage.
        snap,_ = rpc.call('evm_snapshot',[])
        b = D.deploy(rcode, rpc)
        refres = [D.probe(b,s,rpc) for s in sels] if b else None
        rpc.call('evm_revert',[snap])
        a = D.deploy(code, rpc)
        if not b: skip+=1; continue
        if not a:
            print(f'FAIL {src.name}:{name}  creation code does not deploy'); bad+=1; continue
        diff=[(s,D.probe(a,s,rpc),refres[i]) for i,s in enumerate(sels)]
        diff=[d for d in diff if d[1]!=d[2]]
        if diff:
            print(f'DIFF {src.name}:{name}  {len(diff)}/{len(sels)}')
            for s,x,y in diff[:2]: print(f'      {s}: mlir={x[:32]} ref={y[:32]}')
            bad+=1
        else:
            print(f'OK   {src.name}:{name}  {len(sels)} selector(s)'); ok+=1
print(f'\n{ok} agree, {bad} diverge, {skip} skipped')
