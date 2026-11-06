from pathlib import Path
import os


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    import uvicorn
    uvicorn.run('app.server:app', host='127.0.0.1', port=8000)
