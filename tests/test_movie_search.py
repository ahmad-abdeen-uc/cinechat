import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, PropertyMock, patch

import faiss

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.intent_parser import IntentParser, MovieIntent
from src.movie_search import MovieSearch


class TestMovieSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parser = IntentParser()
        cls.addClassCleanup(parser.close)
        cls.parser = Mock(wraps=parser)
        cls.searcher = MovieSearch(parser=cls.parser)
        cls.addClassCleanup(cls.searcher.close)
        cls.db = cls.searcher.engine.conn

    def search(self, query):
        self.parser.reset_mock()
        result = self.searcher.search(query)
        self.parser.parse.assert_called_once_with(query)
        return result

    def search_intent(self, **fields):
        parser = Mock()
        parser.parse.return_value = MovieIntent(**fields)
        result = MovieSearch(parser, self.searcher.engine).search("test request")
        parser.parse.assert_called_once_with("test request")
        return result

    def assert_movies(self, result, sql, parameters=()):
        self.assertEqual(result.status, "ok", result)
        ids = [movie.tmdb_id for movie in result.movies]
        self.assertEqual(len(ids), result.intent.count)
        self.assertEqual(len(ids), len(set(ids)))
        eligible = {row[0] for row in self.db.execute(sql, parameters)}
        self.assertTrue(set(ids) <= eligible, f"Unexpected TMDB IDs: {set(ids) - eligible}")
        for movie in result.movies:
            row = self.db.execute(
                "SELECT title, release_year, vote_average, vote_count, overview "
                "FROM movies WHERE tmdb_id = ?", (movie.tmdb_id,),
            ).fetchone()
            self.assertEqual(tuple(row), (
                movie.title, movie.release_year, movie.rating, movie.vote_count, movie.overview
            ))
            genres = self.db.execute(
                "SELECT genre FROM movie_genres WHERE movie_id = ?", (movie.tmdb_id,),
            )
            self.assertCountEqual(movie.genres, [row[0] for row in genres])
        return ids

    def test_sad_mood_comedy_no_horror(self):
        result = self.search("I'm feeling sad. Recommend funny and uplifting movies, but no horror.")
        self.assertEqual(result.intent.user_mood, "sad")
        self.assertEqual(result.strategy, "semantic")
        self.assert_movies(result, """
            SELECT movie_id FROM movie_genres WHERE genre = 'Comedy'
            EXCEPT SELECT movie_id FROM movie_genres WHERE genre = 'Horror'
        """)

    def test_top_rated_action_after_2010(self):
        # The structured route must work even if FAISS cannot be opened.
        with patch.object(type(self.searcher.engine), "index", new_callable=PropertyMock) as index:
            index.side_effect = AssertionError("Structured search accessed FAISS")
            result = self.search("Top 5 action movies based on audience ratings after 2010.")
        self.assertEqual(result.strategy, "structured")
        self.assertEqual(result.intent.year_min, 2011)
        sql = """
            SELECT m.tmdb_id FROM movies m JOIN movie_genres g ON m.tmdb_id = g.movie_id
            WHERE m.release_year > 2010 AND g.genre = 'Action' AND m.adult IS NOT 1
        """
        ids = self.assert_movies(result, sql)
        average = self.db.execute(
            "SELECT AVG(vote_average) FROM movies WHERE vote_average BETWEEN 0 AND 10 AND vote_count >= 50"
        ).fetchone()[0]
        expected = self.db.execute(
            sql + " ORDER BY m.vote_average IS NOT NULL DESC, "
            "(m.vote_average * COALESCE(m.vote_count, 0) + 250 * ?) "
            "/ (COALESCE(m.vote_count, 0) + 250) DESC, "
            "COALESCE(m.vote_count, 0) DESC, m.tmdb_id LIMIT 5", (average,),
        )
        self.assertEqual(ids, [row[0] for row in expected])

    def test_similar_to_inception(self):
        with patch.object(self.searcher.engine, "encode_query", side_effect=AssertionError(
            "A stored reference embedding should not require an encoder call"
        )):
            result = self.search("Recommend movies like Inception, but no horror.")
        self.assertEqual(result.strategy, "similar")
        ids = self.assert_movies(result, """
            SELECT tmdb_id FROM movies WHERE title <> 'Inception' COLLATE NOCASE
            EXCEPT SELECT movie_id FROM movie_genres WHERE genre = 'Horror'
        """)
        indexed = set(faiss.vector_to_array(self.searcher.engine.index.id_map))
        self.assertTrue(set(ids) <= indexed)

    def test_director(self):
        result = self.search("Show me Christopher Nolan movies released after 2000.")
        self.assert_movies(result, """
            SELECT m.tmdb_id FROM movies m JOIN movie_directors d ON m.tmdb_id = d.movie_id
            WHERE d.director_name = 'Christopher Nolan' AND m.release_year > 2000
        """)

    def test_actor_runtime(self):
        result = self.search("Recommend movies starring Leonardo DiCaprio that are under 120 minutes.")
        self.assert_movies(result, """
            SELECT m.tmdb_id FROM movies m JOIN movie_cast c ON m.tmdb_id = c.movie_id
            WHERE c.actor_name = 'Leonardo DiCaprio' AND m.runtime BETWEEN 1 AND 120
        """)

    def test_action_or_comedy(self):
        result = self.search("I want an Action OR Comedy movie. It does not need to be both.")
        self.assertEqual(result.intent.genre_match, "any")
        self.assert_movies(result, "SELECT movie_id FROM movie_genres WHERE genre IN ('Action', 'Comedy')")

    def test_action_and_comedy(self):
        result = self.search("Recommend movies that are both Action AND Comedy.")
        self.assertEqual(result.intent.genre_match, "all")
        self.assert_movies(result, """
            SELECT movie_id FROM movie_genres WHERE genre = 'Action'
            INTERSECT SELECT movie_id FROM movie_genres WHERE genre = 'Comedy'
        """)

    def test_exclude_titanic(self):
        result = self.search("Recommend romantic movies, but don't recommend Titanic.")
        self.assertIn("Titanic", result.intent.exclude_titles)
        self.assert_movies(result, """
            SELECT movie_id FROM movie_genres WHERE genre = 'Romance'
            EXCEPT SELECT tmdb_id FROM movies
            WHERE title = 'Titanic' COLLATE NOCASE OR original_title = 'Titanic' COLLATE NOCASE
        """)

    def test_impossible_years(self):
        result = self.search("Recommend movies released after 2015 and before 2000.")
        self.assertEqual(result.status, "clarify")
        self.assertTrue(result.clarification_question)
        self.assertFalse(result.movies)

    def test_zero_matching_results(self):
        result = self.search_intent(
            intent="top_rated", genres_include=["Action"], year_min=2100,
            sort_by="weighted_rating", rating_source="audience",
        )
        self.assertEqual(result.intent.year_min, 2100)
        self.assertEqual(result.status, "no_results")
        self.assertTrue(result.warnings)
        self.assertFalse(result.movies)

    def test_unsupported_streaming(self):
        result = self.search("What movies are available on Netflix tonight?")
        self.assertEqual(result.status, "unsupported")
        self.assertTrue(result.warnings)
        self.assertFalse(result.movies)

    def test_duplicate_title_and_year_disambiguation(self):
        # Titanic (1997) has far more votes than the other versions, so search uses it and says so.
        result = self.search_intent(intent="similar_movies", similar_to="Titanic")
        self.assertEqual(result.status, "ok")
        self.assertTrue(any("Titanic (1997)" in warning for warning in result.warnings))
        result = self.search_intent(intent="similar_movies", similar_to="Titanic (1953)")
        self.assertEqual(result.status, "ok")
        result = self.search_intent(intent="similar_movies", similar_to="Titanic (1997)")
        self.assert_movies(result, """
            SELECT tmdb_id FROM movies
            WHERE NOT (title = 'Titanic' COLLATE NOCASE AND release_year = 1997)
        """)

    def test_language_rating_year_and_oldest(self):
        result = self.search_intent(language="Japanese", min_rating=7, year_min=1995,
                                    year_max=2005, sort_by="oldest", count=3)
        self.assert_movies(result, """
            SELECT tmdb_id FROM movies WHERE original_language = 'ja'
            AND vote_average >= 7 AND release_year BETWEEN 1995 AND 2005
        """)
        years = [movie.release_year for movie in result.movies]
        self.assertEqual(years, sorted(years))

    def test_unsupported_hard_constraints(self):
        result = self.search_intent(rating_source="critics", themes_avoid=["grief"])
        self.assertEqual(result.status, "unsupported")
        self.assertEqual(len(result.warnings), 1)
        self.assertFalse(result.movies)

    def test_theme_exclusion_is_a_soft_keyword_filter(self):
        result = self.search_intent(genres_include=["Comedy"], themes_avoid=["violence"], count=10)
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.movies)
        ids = json.dumps([movie.tmdb_id for movie in result.movies])
        flagged = self.searcher.engine.conn.execute(
            "SELECT COUNT(*) FROM movie_keywords WHERE keyword LIKE '%violen%' "
            "AND movie_id IN (SELECT value FROM json_each(?))", (ids,)
        ).fetchone()[0]
        self.assertEqual(flagged, 0)
        self.assertTrue(any("best effort" in warning for warning in result.warnings))

    def test_actor_named_as_director_is_corrected(self):
        result = self.searcher.search_intent("Tom Hanks movies", MovieIntent(director="Tom Hanks"))
        self.assertEqual(result.intent.actor, "Tom Hanks")
        self.assertIsNone(result.intent.director)
        result = self.searcher.search_intent("Christopher Nolan movies", MovieIntent(director="Christopher Nolan"))
        self.assertEqual(result.intent.director, "Christopher Nolan")

    def test_misspelled_reference_title_uses_closest_match(self):
        result = self.search_intent(intent="similar_movies", similar_to="Interstaller")
        self.assertEqual(result.status, "ok")
        self.assertTrue(any("Interstellar" in warning for warning in result.warnings))

    def test_minimum_runtime_cannot_be_silently_ignored(self):
        parser = Mock()
        parser.parse.return_value = MovieIntent()
        result = MovieSearch(parser, self.searcher.engine).search("Find a movie over 120 minutes.")
        self.assertEqual(result.status, "unsupported")
        self.assertIn("Minimum runtime is unavailable.", result.warnings)
        self.assertFalse(result.movies)

    def test_conflict_in_parsed_filters(self):
        result = self.search_intent(genres_include=["Horror"], genres_exclude=["Horror"])
        self.assertEqual(result.status, "clarify")
        self.assertFalse(result.movies)


if __name__ == "__main__":
    unittest.main(verbosity=2)
