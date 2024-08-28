from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from db import get_async_db, engine
from chat import ChatService
from fastapi.middleware.cors import CORSMiddleware
import model as models
from sqlalchemy import select
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/delete")
async def delete_data():
    async with engine.begin() as conn:
        await conn.run_sync(models.Base.metadata.drop_all)
        await conn.run_sync(models.Base.metadata.create_all)
    return {"message": "Data deleted"}


@app.get("/chat/init")
def init_chat():
    chat = models.Chat(message="Welcome to the Human Rights AI Assistant! 🌟\n\nI'm here to help you navigate the complex world of human rights and criminal defense. Whether you have questions about laws, rights, or legal processes, I'm at your service. Remember, while I can provide information and guidance, it's always best to consult with a qualified human lawyer for personalized legal advice.\n\nHow can I assist you today?", role=models.ChatRole.USER)
    return {"message": "Chat initialized"}


@app.get("/chat")
async def read_chat(
    db: AsyncSession = Depends(get_async_db)
):
    stmt = select(models.Chat)
    result = await db.execute(stmt)
    chats = result.scalars().all()
    return chats

@app.post("/chat")
async def create_chat(
    query: str,
    db: AsyncSession = Depends(get_async_db)
):
    chat_service = ChatService(db)
    try:
        response = await chat_service.process_chat(query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9080)