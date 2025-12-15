"""Data loader for Wikipedia Pageviews using Wikimedia REST API."""

import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple
from pathlib import Path
from tqdm import tqdm


class WikipediaPageviewsLoader:
    """
    Loader for Wikipedia pageview data using Wikimedia REST API.

    API Documentation: https://wikitech.wikimedia.org/wiki/Analytics/AQS/Pageviews
    """

    BASE_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"

    def __init__(self, cache_dir: str = "data"):
        """
        Initialize the loader.

        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _build_url(
        self,
        project: str,
        article: str,
        access: str,
        agent: str,
        granularity: str,
        start: str,
        end: str
    ) -> str:
        """Build API URL."""
        # URL encode the article title
        article = article.replace(' ', '_')
        url = f"{self.BASE_URL}/{project}/{access}/{agent}/{article}/{granularity}/{start}/{end}"
        return url

    def _fetch_data(
        self,
        article: str,
        start_date: str,
        end_date: str,
        project: str = "en.wikipedia",
        access: str = "all-access",
        agent: str = "user",
        granularity: str = "daily"
    ) -> pd.DataFrame:
        """
        Fetch pageview data from API.

        Args:
            article: Article title (e.g., "Bitcoin")
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            project: Wikipedia project (default: en.wikipedia)
            access: Access type (default: all-access)
            agent: Agent type (default: user)
            granularity: Time granularity (default: daily)

        Returns:
            DataFrame with columns: timestamp, views
        """
        url = self._build_url(
            project, article, access, agent, granularity,
            start_date, end_date
        )

        headers = {
            'User-Agent': 'TimeSeriesForecastingProject/1.0 (Educational Purpose)'
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            if 'items' not in data:
                raise ValueError(f"No data returned for {article}")

            # Convert to DataFrame
            df = pd.DataFrame(data['items'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d%H')
            df = df[['timestamp', 'views']].rename(columns={'views': 'y'})
            df = df.sort_values('timestamp').reset_index(drop=True)

            return df

        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {e}")

    def _cache_file_path(self, article: str, start_date: str, end_date: str) -> Path:
        """Get cache file path for given parameters."""
        safe_article = article.replace(' ', '_').replace('/', '_')
        filename = f"{safe_article}_{start_date}_{end_date}.json"
        return self.cache_dir / filename

    def _save_cache(self, df: pd.DataFrame, cache_path: Path, metadata: dict):
        """Save data to cache with metadata."""
        cache_data = {
            'metadata': metadata,
            'data': df.to_dict(orient='records')
        }

        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2, default=str)

    def _load_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """Load data from cache if exists."""
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)

            df = pd.DataFrame(cache_data['data'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            return None

    def load_pageviews(
        self,
        article: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
        project: str = "en.wikipedia"
    ) -> pd.DataFrame:
        """
        Load Wikipedia pageview data (with caching).

        Args:
            article: Article title (e.g., "Bitcoin", "Taylor_Swift")
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            use_cache: Whether to use cached data if available
            project: Wikipedia project

        Returns:
            DataFrame with columns: ds (date), y (pageviews)
        """
        # Convert dates to API format (YYYYMMDD00)
        start_api = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d00")
        end_api = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d00")

        # Check cache
        cache_path = self._cache_file_path(article, start_date, end_date)
        if use_cache:
            cached_df = self._load_cache(cache_path)
            if cached_df is not None:
                print(f"[OK] Loaded data from cache: {cache_path}")
                cached_df = cached_df.rename(columns={'timestamp': 'ds'})
                return cached_df

        # Fetch from API
        print(f"Fetching pageviews for '{article}' from {start_date} to {end_date}...")
        df = self._fetch_data(article, start_api, end_api, project=project)

        # Save to cache
        metadata = {
            'article': article,
            'start_date': start_date,
            'end_date': end_date,
            'project': project,
            'downloaded_at': datetime.now().isoformat(),
            'num_records': len(df)
        }
        self._save_cache(df, cache_path, metadata)
        print(f"[OK] Data cached to: {cache_path}")

        # Rename columns to standard names
        df = df.rename(columns={'timestamp': 'ds'})

        return df


def load_data(
    page_title: str,
    start_date: str,
    end_date: str,
    cache_dir: str = "data",
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Convenience function to load Wikipedia pageview data.

    Args:
        page_title: Wikipedia page title
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        cache_dir: Cache directory
        use_cache: Whether to use cache

    Returns:
        DataFrame with columns: ds (date), y (pageviews)
    """
    loader = WikipediaPageviewsLoader(cache_dir=cache_dir)
    df = loader.load_pageviews(
        article=page_title,
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache
    )

    return df


if __name__ == "__main__":
    # Test the loader
    df = load_data(
        page_title="Bitcoin",
        start_date="2020-01-01",
        end_date="2024-12-31"
    )
    print(f"\nLoaded {len(df)} records")
    print(df.head())
    print(df.tail())
    print(f"\nDate range: {df['ds'].min()} to {df['ds'].max()}")
    print(f"Total days: {(df['ds'].max() - df['ds'].min()).days}")
