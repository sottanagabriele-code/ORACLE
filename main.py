from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
import os

app = FastAPI()

@app.get("/")
async def home():
    return HTMLResponse("""
    <html>
        <head>
            <title>ORACLE</title>
        </head>
        <body style="background:black;color:white;font-family:Arial;text-align:center;padding-top:50px;">
            <h1>ORACLE è LIVE 🚀</h1>
            <p>Il backend FastAPI funziona.</p>
        </body>
    </html>
    """)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
