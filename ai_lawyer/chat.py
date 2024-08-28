from sqlalchemy import select
from model import Chat, ChatRole
from finetune import query_data
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import List, Union
from sqlalchemy.ext.asyncio import AsyncSession
import logging

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_all_messages_roles(self):
        stmt = (
            select(Chat)
            .where(Chat.role.in_([ChatRole.ASSISTANT, ChatRole.USER, ChatRole.SYSTEM]))
            .order_by(Chat.created_at.asc())
        )
        result = await self.db.execute(stmt)
        messages = result.scalars().all()
        logger.debug(f"messages: {messages}")
        return list(messages) if messages else None

    async def get_message_history(self):
        message_history: List[Union[HumanMessage, AIMessage, SystemMessage]] = []
        messages = await self.get_all_messages_roles()
        for message in messages:
            if message.role == ChatRole.USER:
                message_history.append(HumanMessage(content=message.message))
            elif message.role == ChatRole.ASSISTANT:
                message_history.append(AIMessage(content=message.message))
            elif message.role == ChatRole.SYSTEM:
                message_history.append(SystemMessage(content=message.message))
        return message_history

    async def process_chat(self, user_message: str):
        try:
            # Add user message to the database
            await self.add_user_message(user_message)

            # Get chat history
            message_history = await self.get_message_history()

            # Add the new user message to the history
            message_history.append(HumanMessage(content=user_message))

            # Query data with chat history
            response = await query_data(message_history)

            # Add assistant response to the database
            await self.add_assistant_message(response)

            return response

        except Exception as e:
            logger.error(f"Error processing chat: {e}")
            raise

    async def add_message(self, role: ChatRole, content: str):
        message = Chat(role=role, message=content)
        self.db.add(message)
        await self.db.commit()
        await self.db.refresh(message)
        return message

    async def add_user_message(self, content: str):
        return await self.add_message(ChatRole.USER, content)

    async def add_assistant_message(self, content: str):
        return await self.add_message(ChatRole.ASSISTANT, content)