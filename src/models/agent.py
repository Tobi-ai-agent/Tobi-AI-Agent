# src/models/agent.py
import numpy as np
import random
from collections import deque
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from PIL import Image

from src.models.network import ConvQNetwork


class DQNAgent:
    """Trading agent using DQN with convolutional networks."""
    
    def __init__(self, state_size, action_size, config, device=None):
        """Initialize the agent.
        
        Args:
            state_size (tuple): Shape of the input state (channels, height, width)
            action_size (int): Number of possible actions
            config (dict): Configuration dictionary
            device (torch.device): Device to run the model on
        """
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
        # Set device (prioritize MPS on Mac if available)
        if device:
            self.device = device
        elif config.get('USE_MPS', False) and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Metal Performance Shaders) device")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("Using CUDA device")
        else:
            self.device = torch.device("cpu")
            print("Using CPU device")
        
        # Initialize Q-Networks (online and target)
        self.qnetwork_local = ConvQNetwork(state_size[0], action_size).to(self.device)
        self.qnetwork_target = ConvQNetwork(state_size[0], action_size).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=config['LEARNING_RATE'])
        
        # Initialize replay memory
        self.memory = deque(maxlen=config['MEMORY_SIZE'])
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        # Initialize preprocessing transform
        self.preprocess = transforms.Compose([
            transforms.Resize(config['IMAGE_SIZE']),
            transforms.ToTensor()
        ])
    
    def step(self, state, action, reward, next_state, done):
        """Add experience to memory and learn if enough samples are available.
        
        Args:
            state (numpy.array): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (numpy.array): Next state
            done (bool): Whether the episode is done
        """
        # Convert images to tensors and add to replay memory
        state_tensor = self._preprocess_state(state)
        next_state_tensor = self._preprocess_state(next_state)
        
        self.memory.append((state_tensor, action, reward, next_state_tensor, done))
        
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % self.config.get('UPDATE_EVERY', 4)
        if self.t_step == 0 and len(self.memory) > self.config['BATCH_SIZE']:
            experiences = random.sample(self.memory, k=self.config['BATCH_SIZE'])
            self._learn(experiences, self.config['GAMMA'])
    
    def act(self, state, epsilon=0.0):
        """Return action based on the current state using epsilon-greedy policy.
        
        Args:
            state (numpy.array): Current state
            epsilon (float): Epsilon for epsilon-greedy action selection
            
        Returns:
            int: Selected action
        """
        # Preprocess state
        state_tensor = self._preprocess_state(state).unsqueeze(0).to(self.device)
        
        # Set the network to evaluation mode
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        # Set the network back to training mode
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def _learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        
        Args:
            experiences (tuple): Tuple of (s, a, r, s', done) tuples
            gamma (float): Discount factor
        """
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        # Convert to tensors
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(1).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).unsqueeze(1).to(self.device)
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self._soft_update(self.qnetwork_local, self.qnetwork_target, self.config.get('TAU', 1e-3))
    
    def _soft_update(self, local_model, target_model, tau):
        """Soft update target network parameters.
        
        θ_target = τ*θ_local + (1 - τ)*θ_target
        
        Args:
            local_model (nn.Module): Source model
            target_model (nn.Module): Target model
            tau (float): Interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


    def _preprocess_state(self, state):
        """Convert state (image) to tensor and preprocess.
        
        Args:
            state (numpy.array): RGB image array
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        if state is None:
            # Return zero tensor if state is None
            return torch.zeros((3, *self.config['IMAGE_SIZE']), dtype=torch.float32)
            
        # Convert to PIL Image if it's a numpy array
        if isinstance(state, np.ndarray):
            image = Image.fromarray(state)
        else:
            image = state
            
        # Ensure image is in RGB mode (3 channels)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Apply preprocessing transforms
        return self.preprocess(image)
    
    def save(self, path):
        """Save model parameters to file.
        
        Args:
            path (str): Path to save the model
        """
        torch.save(self.qnetwork_local.state_dict(), path)
    
    def load(self, path):
        """Load model parameters from file.
        
        Args:
            path (str): Path to load the model from
        """
        self.qnetwork_local.load_state_dict(torch.load(path, map_location=self.device))
        self.qnetwork_target.load_state_dict(torch.load(path, map_location=self.device))