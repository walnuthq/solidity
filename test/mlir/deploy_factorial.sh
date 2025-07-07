#!/bin/bash

# Contract bytecode from solc output
BYTECODE="60808060405234601557610148908161001a8239f35b5f80fdfe60808060405260043610156011575f80fd5b5f3560e01c908163193170f914604d575063de29278914602f575f80fd5b346049575f36600319011260495760205f54604051908152f35b5f80fd5b346049576020366003190112604957600435906039821160ae575060015f5560025b81811115607857005b5f5481810290808204831490151715609a575f555f198114609a57600101606f565b634e487b7160e01b5f52601160045260245ffd5b62461bcd60e51b81526020600482015260156024820152744f766572666c6f773a206e20746f6f206c6172676560581b6044820152606490fdfea264697066735822122077831e18fe9f6dafc6c554743f2b2504ac228bdff886ad8ea4ca30aaa0e0e64164736f6c63782c302e382e33312d646576656c6f702e323032352e382e31312b636f6d6d69742e31313863356338332e6d6f64005d"

# Deploy using cast send
echo "Deploying FactorialStorage contract..."
DEPLOY_TX=$(cast send --rpc-url http://127.0.0.1:8545 \
  --private-key 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80 \
  --create $BYTECODE)

echo "Deploy transaction:"
echo "$DEPLOY_TX"

# Extract the contract address from the output
CONTRACT_ADDR=$(echo "$DEPLOY_TX" | grep "contractAddress" | awk '{print $2}')
echo ""
echo "Contract deployed at: $CONTRACT_ADDR"
echo ""
echo "You can now interact with the contract:"
echo "  - Call computeFactorial(5): cast send $CONTRACT_ADDR 'computeFactorial(uint256)' 5 --rpc-url http://127.0.0.1:8545 --private-key 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
echo "  - Get result: cast call $CONTRACT_ADDR 'getResult()' --rpc-url http://127.0.0.1:8545"
