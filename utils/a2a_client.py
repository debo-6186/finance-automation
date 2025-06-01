"""
Agent2Agent (A2A) Protocol Client Implementation.
Allows agents to communicate with other agents using the A2A protocol.
"""
import asyncio
from typing import Any, Dict, List, Optional, Union
import httpx
import structlog

from .a2a_server import (
    Message, MessageRole, Part, PartType, Task, TaskState, 
    AgentCard, SendTaskRequest, SendTaskResponse,
    create_text_part, create_data_part
)

logger = structlog.get_logger(__name__)

class A2AClient:
    """
    Client for communicating with A2A protocol servers.
    Handles agent discovery, task submission, and response processing.
    """
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        self._agent_cards: Dict[str, AgentCard] = {}
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def discover_agent(self, agent_url: str) -> AgentCard:
        """
        Discover an agent's capabilities by fetching its Agent Card.
        
        Args:
            agent_url: The base URL of the agent
            
        Returns:
            The agent's capabilities as an AgentCard
        """
        try:
            if not agent_url.startswith(('http://', 'https://')):
                agent_url = f"http://{agent_url}"
            
            agent_card_url = f"{agent_url}/.well-known/agent.json"
            logger.info(f"Discovering agent at {agent_card_url}")
            
            response = await self.client.get(agent_card_url)
            response.raise_for_status()
            
            agent_card_data = response.json()
            agent_card = AgentCard(**agent_card_data)
            
            # Cache the agent card
            self._agent_cards[agent_url] = agent_card
            
            logger.info(f"Discovered agent: {agent_card.name} with capabilities: {agent_card.capabilities}")
            return agent_card
            
        except httpx.RequestError as e:
            logger.error(f"Failed to connect to agent at {agent_url}: {e}")
            raise ConnectionError(f"Could not connect to agent at {agent_url}: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error when discovering agent at {agent_url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error discovering agent at {agent_url}: {e}")
            raise
    
    async def send_task(self, agent_url: str, message: Union[str, Message], 
                       task_id: Optional[str] = None) -> Task:
        """
        Send a task to an agent.
        
        Args:
            agent_url: The base URL of the agent
            message: The message to send (string or Message object)
            task_id: Optional task ID for continuing a conversation
            
        Returns:
            The task response from the agent
        """
        try:
            if not agent_url.startswith(('http://', 'https://')):
                agent_url = f"http://{agent_url}"
            
            # Convert string message to Message object if needed
            if isinstance(message, str):
                message_obj = Message(
                    role=MessageRole.USER,
                    parts=[create_text_part(message)]
                )
            else:
                message_obj = message
            
            # Create the request
            request = SendTaskRequest(message=message_obj, task_id=task_id)
            
            # Send the task
            endpoint = f"{agent_url}/tasks/send"
            logger.info(f"Sending task to {endpoint}")
            
            response = await self.client.post(
                endpoint,
                json=request.model_dump(mode='json'),
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            # Parse the response
            response_data = response.json()
            task_response = SendTaskResponse(**response_data)
            
            logger.info(f"Task {task_response.task.id} sent successfully, state: {task_response.task.state}")
            return task_response.task
            
        except httpx.RequestError as e:
            logger.error(f"Failed to send task to agent at {agent_url}: {e}")
            raise ConnectionError(f"Could not send task to agent at {agent_url}: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error when sending task to agent at {agent_url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error sending task to agent at {agent_url}: {e}")
            raise
    
    async def get_task_status(self, agent_url: str, task_id: str) -> Task:
        """
        Get the status of a task.
        
        Args:
            agent_url: The base URL of the agent
            task_id: The ID of the task
            
        Returns:
            The current task state
        """
        try:
            if not agent_url.startswith(('http://', 'https://')):
                agent_url = f"http://{agent_url}"
            
            endpoint = f"{agent_url}/tasks/{task_id}"
            response = await self.client.get(endpoint)
            response.raise_for_status()
            
            task_data = response.json()
            task = Task(**task_data)
            
            return task
            
        except httpx.RequestError as e:
            logger.error(f"Failed to get task status from agent at {agent_url}: {e}")
            raise ConnectionError(f"Could not get task status from agent at {agent_url}: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error when getting task status from agent at {agent_url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting task status from agent at {agent_url}: {e}")
            raise
    
    async def wait_for_task_completion(self, agent_url: str, task_id: str, 
                                      poll_interval: float = 1.0, max_wait: float = 60.0) -> Task:
        """
        Wait for a task to complete by polling its status.
        
        Args:
            agent_url: The base URL of the agent
            task_id: The ID of the task
            poll_interval: How often to poll (seconds)
            max_wait: Maximum time to wait (seconds)
            
        Returns:
            The completed task
        """
        start_time = asyncio.get_event_loop().time()
        
        while True:
            task = await self.get_task_status(agent_url, task_id)
            
            if task.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED]:
                return task
            
            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > max_wait:
                raise TimeoutError(f"Task {task_id} did not complete within {max_wait} seconds")
            
            # Wait before next poll
            await asyncio.sleep(poll_interval)
    
    async def cancel_task(self, agent_url: str, task_id: str) -> Dict[str, str]:
        """
        Cancel a running task.
        
        Args:
            agent_url: The base URL of the agent
            task_id: The ID of the task to cancel
            
        Returns:
            Cancellation status
        """
        try:
            if not agent_url.startswith(('http://', 'https://')):
                agent_url = f"http://{agent_url}"
            
            endpoint = f"{agent_url}/tasks/{task_id}/cancel"
            response = await self.client.post(endpoint)
            response.raise_for_status()
            
            return response.json()
            
        except httpx.RequestError as e:
            logger.error(f"Failed to cancel task at agent {agent_url}: {e}")
            raise ConnectionError(f"Could not cancel task at agent {agent_url}: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error when canceling task at agent {agent_url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error canceling task at agent {agent_url}: {e}")
            raise
    
    def get_cached_agent_card(self, agent_url: str) -> Optional[AgentCard]:
        """Get a cached agent card if available."""
        return self._agent_cards.get(agent_url)
    
    async def send_text_message(self, agent_url: str, text: str, task_id: Optional[str] = None) -> str:
        """
        Send a simple text message and return the text response.
        
        Args:
            agent_url: The base URL of the agent
            text: The text message to send
            task_id: Optional task ID for continuing a conversation
            
        Returns:
            The text response from the agent
        """
        task = await self.send_task(agent_url, text, task_id)
        
        # Wait for completion if needed
        if task.state == TaskState.WORKING:
            task = await self.wait_for_task_completion(agent_url, task.id)
        
        if task.state == TaskState.FAILED:
            raise RuntimeError(f"Task failed: {task.error}")
        
        # Extract text response from the last agent message
        for message in reversed(task.messages):
            if message.role == MessageRole.AGENT:
                for part in message.parts:
                    if part.type == PartType.TEXT:
                        return part.content
        
        return "No response received"
    
    async def send_data_message(self, agent_url: str, data: Dict[str, Any], 
                               task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Send a structured data message and return the data response.
        
        Args:
            agent_url: The base URL of the agent
            data: The data to send
            task_id: Optional task ID for continuing a conversation
            
        Returns:
            The data response from the agent
        """
        message = Message(
            role=MessageRole.USER,
            parts=[create_data_part(data)]
        )
        
        task = await self.send_task(agent_url, message, task_id)
        
        # Wait for completion if needed
        if task.state == TaskState.WORKING:
            task = await self.wait_for_task_completion(agent_url, task.id)
        
        if task.state == TaskState.FAILED:
            raise RuntimeError(f"Task failed: {task.error}")
        
        # Extract data response from the last agent message
        for message in reversed(task.messages):
            if message.role == MessageRole.AGENT:
                for part in message.parts:
                    if part.type == PartType.DATA:
                        return part.content
        
        return {} 