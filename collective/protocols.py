import asyncio
from abc import ABC
from collections import deque
from typing import ClassVar

from lionagi import Session
from lionagi.core.action.action_manager import ActionManager, Tool
from lionagi.core.communication.action_request import ActionRequest
from lionagi.core.communication.action_response import ActionResponse
from lionagi.core.communication.message import RoledMessage
from lionagi.core.generic.node import Node
from lionagi.protocols.operatives.action import ActionResponseModel
from pydantic import BaseModel, Field, PrivateAttr

__all__ = (
    "Ability",
    "Capability",
    "Recipient",
    "Agent",
)


RESERVED_METHODS = (
    "__init__",
    "actions",
    "generate_tools",
)


class Ability(BaseModel):

    _tool: ClassVar[Tool | None] = None

    @classmethod
    def func_callable(cls, *args, **kwargs):
        """
        Return a function that does the actual work.
        Must be overridden in concrete classes.
        """

        async def some_function(*args, **kwargs) -> BaseModel:
            pass

        return some_function

    @classmethod
    def as_tool(cls) -> Tool:
        """
        Convert this Ability into a Tool for registration with ActionManager.
        """
        if cls._tool is None:
            func = cls.func_callable()
            tool = Tool(function=func)
            cls._tool = tool
        return cls._tool


class Capability(ABC):
    """
    A Capability groups multiple Abilities (Tools) under one manager.
    Each Capability can process ActionRequests directed to its abilities.
    """

    default_abilities: ClassVar[list[type[Ability]]] = []

    def __init__(self):
        self.ability_registry = {}
        self.action_manager = ActionManager()

    def register_ability(self, ability: type[Ability]):
        tool = ability.as_tool()
        func_name = tool.function.__name__
        self.action_manager.register_tool(tool, update=True)
        self.ability_registry[func_name] = ability

    def dump_log(self):
        self.action_manager.logger.dump()

    async def process(self, action_request: ActionRequest) -> ActionResponse:
        """
        Override if you want advanced logic before/after invocation.
        By default, just invoke the tool and wrap the response.
        """
        response = await self.action_manager.invoke(action_request)
        return ActionResponseModel(
            function=action_request.function,
            arguments=action_request.arguments,
            output=response,
        )

    @classmethod
    def create(cls):
        """
        Factory method that registers all default_abilities.
        """
        capability = cls()
        for ability_cls in cls.default_abilities:
            capability.register_ability(ability_cls)
        return capability


class Recipient(BaseModel):
    name: str


class Agent(Node):

    session: Session | None = Field(default_factory=Session)

    mailbox: dict[str, dict[str, deque[RoledMessage]]] = {
        "pending_in": deque(),
        "pending_out": {},
    }

    capabilities: dict[str, Capability] = Field(default_factory=dict)
    _execute_mode: bool = PrivateAttr(default=False)

    @property
    def name(self): ...

    def register_capability(self, capability: type[Capability]):
        capability = capability.create()
        self.capabilities[capability.__class__.__name__] = capability

    def send(self, message: RoledMessage):
        if not self.mailbox["pending_out"].get(message.recipient, None):
            self.mailbox["pending_out"][message.recipient] = deque()
        self.mailbox["pending_out"][message.recipient].append(message)

    async def execute(self):
        self._execute_mode = True
        while self.mailbox["pending_in"]:
            await self.forward()
        self._execute_mode = False

    async def forward(self):
        messages = self.mailbox["pending_in"].popleft()
        response = await self.process(messages)
        self.send(response)

    async def _decide_recipient(
        self, message: ActionResponse, available_recipients: list[str]
    ) -> str:
        recipient = await self.session.default_branch.operate(
            instruction="please decide a the recipient for our message",
            context=message.content.to_dict(),
            guidance=f"must choose one of the following: {available_recipients}",
            operative_model=Recipient,
        )
        return recipient.name

    async def process(
        self, action_request: ActionRequest, available_recipients: list[str]
    ):
        """
        The simplest approach:
        1) find the right Capability that can handle the function
        2) call its .process()
        """

        # Search each capability to see if it has the requested tool
        for cap_name, cap_obj in self.capabilities.items():
            if action_request.function in cap_obj.action_manager.registry:
                action_response_model = await cap_obj.process(action_request)
                action_response = ActionResponse(
                    action_request=action_request,
                    output=action_response_model.output,
                )
                action_response.sender = self.name
                action_response.recipient = await self._decide_recipient(
                    action_response, available_recipients=available_recipients
                )
                self.send(action_response)
                return
        raise ValueError(
            f"No capability can handle function '{action_request.function}'!"
        )

    def join_orchestrator(self, orchestrator):
        orchestrator.agents[self.session.ln_id] = self


class Orchestrator:

    def __init__(self, refresh_rate: int = 60, agents: list[Agent] = []):
        self.agents: dict[str, Agent] = {}
        self.mailbox = {
            "pending_in": deque(),
            "pending_out": {agent.name: deque() for agent in agents},
            "external_out": deque(),
        }
        self.external_sources = {}
        self._stop_event = asyncio.Event()
        self._refresh_rate = refresh_rate
        self._lock = asyncio.Lock()
        self._execute_mode = False

    def collect(self):
        for agent in self.agents.values():
            for k, v in agent.mailbox["pending_out"].items():
                self.mailbox["pending_in"].extend(v)
            self.mailbox["pending_in"].extend(agent.mailbox["pending_out"])

    def send(self):
        while self.mailbox["pending_in"]:
            msg = self.mailbox["pending_in"].popleft()
            if msg.recipient in self.agents:
                self.agents[msg.recipient].mailbox["pending_in"].append(msg)
            else:
                self.mailbox["external_out"].append(msg)

    async def forward(self):
        self.collect()
        self.send()
        tasks = [asyncio.create_task(agent.execute()) for agent in self.agents.values()]
        asyncio.gather(*tasks)
        self.collect()
        await asyncio.sleep(0.1)

    async def stop(self):
        self._stop_event.set()

    async def execute(self):
        self._execute_mode = True
        while not self._stop_event.is_set():
            await self.forward()
            await asyncio.sleep(self._refresh_rate)

        self._execute_mode = False
