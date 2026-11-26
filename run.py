from pathlib import Path
import logging
import os


def build_semantic_map_if_missing():
    from recommender.build_occupation_map import INDEX, OUTPUT, build

    if OUTPUT.exists() or not (INDEX / 'manifest.json').is_file():
        return

    try:
        build()
    except Exception:
        logging.getLogger(__name__).warning(
            'Could not build the optional semantic map; starting the app without it.',
            exc_info=True,
        )


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    import uvicorn

    build_semantic_map_if_missing()
    uvicorn.run('app.server:app', host='127.0.0.1', port=8000)
