
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.recommendation_engine import RecommendationEngine

engine = RecommendationEngine()

test_cases = [
    {
        "query": (
            "Mind-bending science fiction movies "
            "about dreams and reality"
        )
    },
    {
        "query": (
            "Emotional movies about friendship, "
            "love, and finding happiness"
        )
    },
    {
        "query": (
            "Space exploration, astronauts, "
            "distant planets and survival"
        ),
        "exclude_genres": ["Documentary"]
    },
    {
        "query": (
            "Funny and heartwarming movies "
            "that can cheer someone up"
        )
    },
]

try:
    for test in test_cases:
        print("\n" + "=" * 75)
        print("QUERY:", test["query"])
        print("=" * 75)

        recommendations = engine.recommend(
            query=test["query"],
            top_k=5,
            exclude_genres=test.get("exclude_genres"),
        )

        for rank, movie in enumerate(
            recommendations, start=1
        ):
            print(
                f"\n{rank}. {movie['title']} "
                f"({movie['release_year']})"
            )
            print(f"   TMDB ID: {movie['tmdb_id']}")
            print(f"   Rating: {movie['vote_average']}")
            print(f"   Votes: {movie['vote_count']}")
            print(
                f"   Similarity: "
                f"{movie['similarity']:.4f}"
            )
            print(
                f"   Final Score: "
                f"{movie['recommendation_score']:.4f}"
            )

    print("\nHybrid recommendation test completed.")

finally:
    engine.close()
