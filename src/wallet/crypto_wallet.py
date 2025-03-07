#!/usr/bin/env python
# src/wallet/crypto_wallet.py
import os
import json
import logging
import getpass
import base64
from pathlib import Path
import hashlib
from cryptography.fernet import Fernet
from abc import ABC, abstractmethod

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CryptoWallet")

class CryptoWallet(ABC):
    """Base abstract class for cryptocurrency wallets."""
    
    def __init__(self, wallet_dir='wallets', wallet_name=None):
        """Initialize wallet.
        
        Args:
            wallet_dir (str): Directory to store wallet files
            wallet_name (str): Name of the wallet
        """
        self.wallet_dir = Path(wallet_dir)
        self.wallet_dir.mkdir(exist_ok=True, parents=True)
        
        self.wallet_name = wallet_name or self.__class__.__name__.lower()
        self.wallet_path = self.wallet_dir / f"{self.wallet_name}.json"
        self.wallet_data = {}
        self._key = None

    def _generate_key(self, password):
        """Generate encryption key from password."""
        return hashlib.sha256(password.encode()).digest()
    
    def _encrypt_data(self, data, password):
        """Encrypt wallet data with password."""
        key = self._generate_key(password)
        f = Fernet(base64.urlsafe_b64encode(key))
        return f.encrypt(json.dumps(data).encode()).decode()
    
    def _decrypt_data(self, encrypted_data, password):
        """Decrypt wallet data with password."""
        try:
            key = self._generate_key(password)
            f = Fernet(base64.urlsafe_b64encode(key))
            decrypted = f.decrypt(encrypted_data.encode()).decode()
            return json.loads(decrypted)
        except Exception as e:
            logger.error(f"Failed to decrypt wallet: {e}")
            return None
    
    def create_wallet(self, password=None):
        """Create a new wallet with optional password protection.
        
        Args:
            password (str, optional): Password for encryption
        
        Returns:
            dict: Wallet data
        """
        if self.wallet_path.exists():
            logger.warning(f"Wallet already exists at {self.wallet_path}")
            return self.load_wallet(password)
        
        # Get password if not provided
        if password is None:
            password = getpass.getpass("Enter new wallet password: ")
            confirm = getpass.getpass("Confirm password: ")
            if password != confirm:
                logger.error("Passwords do not match")
                return None
        
        # Generate new wallet
        self.wallet_data = self._generate_wallet()
        
        # Save wallet
        self._save_wallet(password)
        
        return self.wallet_data
    
    def load_wallet(self, password=None):
        """Load wallet from file.
        
        Args:
            password (str, optional): Password for decryption
        
        Returns:
            dict: Wallet data
        """
        if not self.wallet_path.exists():
            logger.error(f"Wallet does not exist at {self.wallet_path}")
            return None
        
        # Get password if not provided and wallet is encrypted
        wallet_json = json.loads(self.wallet_path.read_text())
        if "encrypted" in wallet_json and wallet_json["encrypted"] and password is None:
            password = getpass.getpass("Enter wallet password: ")
        
        # Load wallet data
        try:
            if "encrypted" in wallet_json and wallet_json["encrypted"]:
                decrypted_data = self._decrypt_data(wallet_json["data"], password)
                if decrypted_data is None:
                    logger.error("Invalid password")
                    return None
                self.wallet_data = decrypted_data
            else:
                self.wallet_data = wallet_json
            
            logger.info(f"Loaded wallet from {self.wallet_path}")
            return self.wallet_data
        except Exception as e:
            logger.error(f"Failed to load wallet: {e}")
            return None
    
    def _save_wallet(self, password=None):
        """Save wallet to file.
        
        Args:
            password (str, optional): Password for encryption
        """
        try:
            # Encrypt wallet if password provided
            if password:
                encrypted_data = self._encrypt_data(self.wallet_data, password)
                wallet_json = {
                    "encrypted": True,
                    "data": encrypted_data
                }
            else:
                wallet_json = self.wallet_data
                wallet_json["encrypted"] = False
            
            # Save to file
            with open(self.wallet_path, 'w') as f:
                json.dump(wallet_json, f, indent=2)
            
            logger.info(f"Saved wallet to {self.wallet_path}")
        except Exception as e:
            logger.error(f"Failed to save wallet: {e}")
    
    @abstractmethod
    def _generate_wallet(self):
        """Generate a new wallet."""
        pass
    
    @abstractmethod
    def get_address(self):
        """Get wallet address."""
        pass
    
    @abstractmethod
    def get_balance(self):
        """Get wallet balance."""
        pass
    
    @abstractmethod
    def sign_transaction(self, transaction):
        """Sign a transaction.
        
        Args:
            transaction: Transaction data
            
        Returns:
            str: Signed transaction
        """
        pass

# Import required libraries for implementation
import base64

try:
    from eth_account import Account
    from web3 import Web3
    ETH_AVAILABLE = True
except ImportError:
    ETH_AVAILABLE = False
    logger.warning("Ethereum support not available. Install web3 and eth_account.")

# Try multiple ways to import Solana modules
SOLANA_AVAILABLE = False
SOLANA_IMPORT_METHOD = None

# Method 1: Try the standard solana import paths
try:
    from solana.keypair import Keypair
    from solana.publickey import PublicKey
    from solana.rpc.api import Client
    SOLANA_AVAILABLE = True
    SOLANA_IMPORT_METHOD = "standard"
    logger.info("Using standard Solana package imports")
except ImportError:
    pass

# Method 2: Try the solders package (sometimes included with solana)
if not SOLANA_AVAILABLE:
    try:
        import solders.keypair as Keypair
        import solders.pubkey as PublicKey
        from solana.rpc.api import Client  # Still try this import
        SOLANA_AVAILABLE = True
        SOLANA_IMPORT_METHOD = "solders"
        logger.info("Using solders package for Solana functionality")
    except ImportError:
        pass

# Method 3: If solana module exists but has a different structure
if not SOLANA_AVAILABLE:
    try:
        import solana
        import inspect
        
        # Debug: Print the solana module structure
        logger.info(f"Solana module found at: {solana.__file__}")
        logger.info(f"Solana module contents: {dir(solana)}")
        
        # Look for submodules that might contain keypair
        for module_name in dir(solana):
            if not module_name.startswith('_'):
                try:
                    module = getattr(solana, module_name)
                    if hasattr(module, 'Keypair'):
                        Keypair = getattr(module, 'Keypair')
                        logger.info(f"Found Keypair in solana.{module_name}")
                    if hasattr(module, 'PublicKey'):
                        PublicKey = getattr(module, 'PublicKey')
                        logger.info(f"Found PublicKey in solana.{module_name}")
                except:
                    pass
        
        # If we found both Keypair and PublicKey
        if 'Keypair' in locals() and 'PublicKey' in locals():
            # Try to find a Client class
            for module_name in dir(solana):
                if not module_name.startswith('_'):
                    try:
                        module = getattr(solana, module_name)
                        if hasattr(module, 'Client'):
                            Client = getattr(module, 'Client')
                            logger.info(f"Found Client in solana.{module_name}")
                            break
                    except:
                        pass
            
            SOLANA_AVAILABLE = True
            SOLANA_IMPORT_METHOD = "dynamic"
            logger.info("Using dynamically discovered Solana imports")
    except ImportError:
        pass

if not SOLANA_AVAILABLE:
    logger.warning("Solana support not available. Install solana package with: pip install solana solders")

class EthereumWallet(CryptoWallet):
    """Ethereum wallet implementation."""
    
    def __init__(self, wallet_dir='wallets', wallet_name='ethereum', provider_url=None):
        """Initialize Ethereum wallet.
        
        Args:
            wallet_dir (str): Directory to store wallet files
            wallet_name (str): Name of the wallet
            provider_url (str): Ethereum provider URL
        """
        super().__init__(wallet_dir, wallet_name)
        
        if not ETH_AVAILABLE:
            raise ImportError("Ethereum support not available. Install web3 and eth_account.")
        
        # Initialize web3
        self.provider_url = provider_url or "https://mainnet.infura.io/v3/your-infura-key"
        self.w3 = Web3(Web3.HTTPProvider(self.provider_url))
    
    def _generate_wallet(self):
        """Generate a new Ethereum wallet."""
        Account.enable_unaudited_hdwallet_features()
        acct, mnemonic = Account.create_with_mnemonic()
        
        return {
            "type": "ethereum",
            "address": acct.address,
            "private_key": acct.key.hex(),
            "mnemonic": mnemonic
        }
    
    def import_wallet(self, private_key=None, mnemonic=None, password=None):
        """Import an existing Ethereum wallet.
        
        Args:
            private_key (str, optional): Private key
            mnemonic (str, optional): Mnemonic phrase
            password (str, optional): Password for encryption
            
        Returns:
            dict: Wallet data
        """
        if private_key:
            acct = Account.from_key(private_key)
            self.wallet_data = {
                "type": "ethereum",
                "address": acct.address,
                "private_key": acct.key.hex()
            }
        elif mnemonic:
            Account.enable_unaudited_hdwallet_features()
            acct = Account.from_mnemonic(mnemonic)
            self.wallet_data = {
                "type": "ethereum",
                "address": acct.address,
                "private_key": acct.key.hex(),
                "mnemonic": mnemonic
            }
        else:
            logger.error("Either private_key or mnemonic must be provided")
            return None
        
        # Save wallet
        self._save_wallet(password)
        
        return self.wallet_data
    
    def get_address(self):
        """Get wallet address."""
        return self.wallet_data.get("address")
    
    def get_balance(self, token_address=None):
        """Get wallet balance.
        
        Args:
            token_address (str, optional): ERC20 token address
            
        Returns:
            float: Balance in ETH or token
        """
        if not self.wallet_data:
            logger.error("Wallet not loaded")
            return None
        
        address = self.wallet_data["address"]
        
        try:
            if token_address:
                # ERC20 token balance
                abi = [
                    {
                        "constant": True,
                        "inputs": [{"name": "_owner", "type": "address"}],
                        "name": "balanceOf",
                        "outputs": [{"name": "balance", "type": "uint256"}],
                        "type": "function"
                    },
                    {
                        "constant": True,
                        "inputs": [],
                        "name": "decimals",
                        "outputs": [{"name": "", "type": "uint8"}],
                        "type": "function"
                    }
                ]
                
                token_contract = self.w3.eth.contract(address=token_address, abi=abi)
                balance = token_contract.functions.balanceOf(address).call()
                decimals = token_contract.functions.decimals().call()
                
                return balance / (10 ** decimals)
            else:
                # ETH balance
                balance_wei = self.w3.eth.get_balance(address)
                return self.w3.from_wei(balance_wei, 'ether')
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return None
    
    def sign_transaction(self, transaction):
        """Sign an Ethereum transaction.
        
        Args:
            transaction (dict): Transaction data with keys:
                - to: Recipient address
                - value: Amount in ETH
                - gas: Gas limit
                - gasPrice: Gas price
                - nonce: Transaction nonce
                - data: Transaction data
                
        Returns:
            str: Signed transaction hash
        """
        if not self.wallet_data:
            logger.error("Wallet not loaded")
            return None
        
        private_key = self.wallet_data["private_key"]
        
        try:
            # Convert ETH to Wei if value is in ETH
            if "value" in transaction and isinstance(transaction["value"], (int, float)):
                transaction["value"] = self.w3.to_wei(transaction["value"], 'ether')
            
            # Sign transaction
            signed_tx = self.w3.eth.account.sign_transaction(transaction, private_key)
            
            return signed_tx.rawTransaction.hex()
        except Exception as e:
            logger.error(f"Failed to sign transaction: {e}")
            return None

class SolanaWallet(CryptoWallet):
    """Solana wallet implementation."""
    
    def __init__(self, wallet_dir='wallets', wallet_name='solana', rpc_url=None):
        """Initialize Solana wallet.
        
        Args:
            wallet_dir (str): Directory to store wallet files
            wallet_name (str): Name of the wallet
            rpc_url (str): Solana RPC URL
        """
        super().__init__(wallet_dir, wallet_name)
        
        if not SOLANA_AVAILABLE:
            raise ImportError("Solana support not available. Install solana-py.")
        
        # Initialize Solana client
        self.rpc_url = rpc_url or "https://api.mainnet-beta.solana.com"
        self.client = Client(self.rpc_url)
        
        # Store import method for reference
        self.import_method = SOLANA_IMPORT_METHOD
        logger.info(f"Using Solana import method: {self.import_method}")
    
    def _generate_wallet(self):
        """Generate a new Solana wallet."""
        try:
            # Handle different keypair instantiation based on import method
            if self.import_method == "standard":
                keypair = Keypair.generate() if hasattr(Keypair, 'generate') else Keypair()
                public_key = str(keypair.public_key)
                secret_key = keypair.secret_key
            elif self.import_method == "solders":
                keypair = Keypair.new_unique()
                public_key = str(keypair.pubkey())
                secret_key = keypair.secret()
            else:
                # Dynamic method - will need to inspect the object
                if hasattr(Keypair, 'generate'):
                    keypair = Keypair.generate()
                else:
                    keypair = Keypair()
                
                # Determine public key attribute
                if hasattr(keypair, 'public_key'):
                    public_key = str(keypair.public_key)
                elif hasattr(keypair, 'pubkey'):
                    public_key = str(keypair.pubkey())
                else:
                    logger.error("Unknown Keypair structure: Can't find public key")
                    public_key = "unknown"
                
                # Determine secret key attribute
                if hasattr(keypair, 'secret_key'):
                    secret_key = keypair.secret_key
                elif hasattr(keypair, 'secret'):
                    secret_key = keypair.secret()
                else:
                    logger.error("Unknown Keypair structure: Can't find secret key")
                    secret_key = b"unknown"
            
            return {
                "type": "solana",
                "address": public_key,
                "private_key": base64.b64encode(secret_key).decode('ascii') if isinstance(secret_key, bytes) else str(secret_key)
            }
        except Exception as e:
            logger.error(f"Failed to generate Solana wallet: {e}")
            # Return a basic structure so at least it doesn't crash
            return {
                "type": "solana",
                "address": "error",
                "private_key": "error generating wallet"
            }
    
    def import_wallet(self, private_key=None, password=None):
        """Import an existing Solana wallet.
        
        Args:
            private_key (str or bytes): Private key (base58, base64, or byte array)
            password (str, optional): Password for encryption
            
        Returns:
            dict: Wallet data
        """
        try:
            if isinstance(private_key, str):
                # Try to handle different private key formats
                try:
                    import base58
                    if len(private_key) == 88:  # Base58 encoded
                        private_key = base58.b58decode(private_key)[:32]
                    else:  # Assume base64 encoded
                        private_key = base64.b64decode(private_key)
                except:
                    # If base58 not available, try base64
                    private_key = base64.b64decode(private_key)
            
            # Handle keypair creation based on import method
            if self.import_method == "standard":
                keypair = Keypair.from_secret_key(private_key)
                public_key = str(keypair.public_key)
                secret_key = keypair.secret_key
            elif self.import_method == "solders":
                keypair = Keypair.from_bytes(private_key)
                public_key = str(keypair.pubkey())
                secret_key = keypair.secret()
            else:
                # Dynamic method - determine the right function to call
                if hasattr(Keypair, 'from_secret_key'):
                    keypair = Keypair.from_secret_key(private_key)
                elif hasattr(Keypair, 'from_bytes'):
                    keypair = Keypair.from_bytes(private_key)
                else:
                    logger.error("Unknown Keypair structure: Can't import private key")
                    raise ImportError("Can't determine how to import private key")
                
                # Determine public key
                if hasattr(keypair, 'public_key'):
                    public_key = str(keypair.public_key)
                elif hasattr(keypair, 'pubkey'):
                    public_key = str(keypair.pubkey())
                else:
                    raise AttributeError("Can't find public key attribute")
                
                # Determine secret key
                if hasattr(keypair, 'secret_key'):
                    secret_key = keypair.secret_key
                elif hasattr(keypair, 'secret'):
                    secret_key = keypair.secret()
                else:
                    raise AttributeError("Can't find secret key attribute")
            
            self.wallet_data = {
                "type": "solana",
                "address": public_key,
                "private_key": base64.b64encode(secret_key).decode('ascii') if isinstance(secret_key, bytes) else str(secret_key)
            }
            
            # Save wallet
            self._save_wallet(password)
            
            return self.wallet_data
            
        except Exception as e:
            logger.error(f"Failed to import wallet: {e}")
            return None
    
    def get_address(self):
        """Get wallet address."""
        return self.wallet_data.get("address")
    
    def get_balance(self, token_address=None):
        """Get wallet balance.
        
        Args:
            token_address (str, optional): SPL token address
            
        Returns:
            float: Balance in SOL or token
        """
        if not self.wallet_data:
            logger.error("Wallet not loaded")
            return None
        
        address = self.wallet_data["address"]
        
        try:
            if token_address:
                logger.info("SPL token balance retrieval not implemented in this version")
                return 0.0
            else:
                # SOL balance
                response = self.client.get_balance(address)
                if 'result' in response and 'value' in response['result']:
                    balance = response['result']['value']
                    return balance / 10**9  # Convert lamports to SOL
                else:
                    logger.error(f"Unexpected response format: {response}")
                    return 0.0
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return None
    
    def _get_keypair(self):
        """Get Solana keypair from wallet data.
        
        Returns:
            Keypair: Solana keypair
        """
        try:
            private_key = base64.b64decode(self.wallet_data["private_key"])
            
            # Handle keypair creation based on import method
            if self.import_method == "standard":
                return Keypair.from_secret_key(private_key)
            elif self.import_method == "solders":
                return Keypair.from_bytes(private_key)
            else:
                # Dynamic method
                if hasattr(Keypair, 'from_secret_key'):
                    return Keypair.from_secret_key(private_key)
                elif hasattr(Keypair, 'from_bytes'):
                    return Keypair.from_bytes(private_key)
                else:
                    raise NotImplementedError("Can't determine how to create keypair from private key")
        except Exception as e:
            logger.error(f"Failed to get keypair: {e}")
            raise
    
    def sign_transaction(self, transaction):
        """Sign a Solana transaction.
        
        Args:
            transaction: Transaction to sign
            
        Returns:
            Transaction: Signed transaction
        """
        if not self.wallet_data:
            logger.error("Wallet not loaded")
            return None
        
        try:
            keypair = self._get_keypair()
            transaction.sign([keypair])
            return transaction
        except Exception as e:
            logger.error(f"Failed to sign transaction: {e}")
            return None

# Simple self-test when run directly
if __name__ == "__main__":
    logger.info("Testing cryptocurrency wallets")
    
    # Test Ethereum wallet
    if ETH_AVAILABLE:
        try:
            eth_wallet = EthereumWallet()
            eth_data = eth_wallet.create_wallet("test123")
            
            if eth_data:
                logger.info(f"Created Ethereum wallet: {eth_wallet.get_address()}")
        except Exception as e:
            logger.error(f"Ethereum wallet test failed: {e}")
    else:
        logger.warning("Skipping Ethereum wallet test - not available")
    
    # Test Solana wallet
    if SOLANA_AVAILABLE:
        try:
            sol_wallet = SolanaWallet()
            sol_data = sol_wallet.create_wallet("test123")
            
            if sol_data:
                logger.info(f"Created Solana wallet: {sol_wallet.get_address()}")
                logger.info(f"Using import method: {sol_wallet.import_method}")
        except Exception as e:
            logger.error(f"Solana wallet test failed: {e}")
    else:
        logger.warning("Skipping Solana wallet test - not available")