"""Occupation-ranking API, independent of training, evaluation and terminal I/O."""

__all__ = ["RankingPipeline"]


def __getattr__(name):
    if name == "RankingPipeline":
        from recommender.models.ranker import RankingPipeline
        return RankingPipeline
    raise AttributeError(name)
