import streamlit as st
import json
from datetime import datetime, timedelta

# --- Constants ---

TOP_50_US_METROS = [
    "Nationwide",
    "New York, NY",
    "Los Angeles, CA",
    "Chicago, IL",
    "Dallas-Fort Worth, TX",
    "Houston, TX",
    "Washington, DC",
    "Miami, FL",
    "Philadelphia, PA",
    "Atlanta, GA",
    "Phoenix, AZ",
    "Boston, MA",
    "San Francisco, CA",
    "Riverside-San Bernardino, CA",
    "Detroit, MI",
    "Seattle, WA",
    "Minneapolis-St. Paul, MN",
    "San Diego, CA",
    "Tampa, FL",
    "Denver, CO",
    "St. Louis, MO",
    "Baltimore, MD",
    "Charlotte, NC",
    "Orlando, FL",
    "San Antonio, TX",
    "Portland, OR",
    "Sacramento, CA",
    "Austin, TX",
    "Pittsburgh, PA",
    "Las Vegas, NV",
    "Cincinnati, OH",
    "Kansas City, MO",
    "Columbus, OH",
    "Indianapolis, IN",
    "Cleveland, OH",
    "San Jose, CA",
    "Nashville, TN",
    "Virginia Beach, VA",
    "Providence, RI",
    "Milwaukee, WI",
    "Jacksonville, FL",
    "Oklahoma City, OK",
    "Raleigh, NC",
    "Memphis, TN",
    "Richmond, VA",
    "New Orleans, LA",
    "Louisville, KY",
    "Salt Lake City, UT",
    "Hartford, CT",
    "Buffalo, NY",
    "Birmingham, AL"
]

# Map some metros to relevant Reddit subreddits for "social" mode
METRO_REDDIT_MAP = {
    "New York, NY": ["nyc", "AskNYC", "newyorkcity", "NewYork"],
    "Los Angeles, CA": ["LosAngeles", "AskLosAngeles", "LAlist"],
    "Chicago, IL": ["chicago"],
    "Dallas-Fort Worth, TX": ["Dallas", "FortWorth", "DFW"],
    "Houston, TX": ["houston"],
    "Washington, DC": ["washingtondc", "nova", "maryland", "DMV"],
    "Miami, FL": ["Miami", "MiamiBeach", "southflorida"],
    "Philadelphia, PA": ["philadelphia", "philly"],
    "Atlanta, GA": ["Atlanta"],
    "Phoenix, AZ": ["phoenix", "AskPhoenix"],
    "Boston, MA": ["boston"],
    "San Francisco, CA": ["sanfrancisco", "bayarea", "AskSF"],
    "Riverside-San Bernardino, CA": ["InlandEmpire", "RiversideCA", "SanBernardino"],
    "Detroit, MI": ["Detroit"],
    "Seattle, WA": ["Seattle"],
    "Minneapolis-St. Paul, MN": ["minneapolis", "SaintPaul", "TwinCities"],
    "San Diego, CA": ["sandiego"],
    "Tampa, FL": ["tampa", "TampaBay"],
    "Denver, CO": ["Denver", "Colorado"],
    "St. Louis, MO": ["StLouis"],
    "Baltimore, MD": ["baltimore"],
    "Charlotte, NC": ["Charlotte", "CLT"],
    "Orlando, FL": ["orlando"],
    "San Antonio, TX": ["sanantonio"],
    "Portland, OR": ["Portland"],
    "Sacramento, CA": ["Sacramento"],
    "Austin, TX": ["Austin"],
    "Pittsburgh, PA": ["pittsburgh"],
    "Las Vegas, NV": ["lasvegas"],
    "Cincinnati, OH": ["cincinnati"],
    "Kansas City, MO": ["kansascity"],
    "Columbus, OH": ["Columbus"],
    "Indianapolis, IN": ["indianapolis"],
    "Cleveland, OH": ["Cleveland"],
    "San Jose, CA": ["SanJose", "bayarea"],
    "Nashville, TN": ["nashville"],
    "Virginia Beach, VA": ["virginiabeach", "norfolk", "hamptonroads"],
    "Providence, RI": ["rhodeisland", "Providence"],
    "Milwaukee, WI": ["milwaukee"],
    "Jacksonville, FL": ["jacksonville"],
    "Oklahoma City, OK": ["oklahomacity", "okc"],
    "Raleigh, NC": ["raleigh", "triangle"],
    "Memphis, TN": ["memphis"],
    "Richmond, VA": ["richmond"],
    "New Orleans, LA": ["neworleans", "NOLA"],
    "Louisville, KY": ["louisville"],
    "Salt Lake City, UT": ["SaltLakeCity", "Utah"],
    "Hartford, CT": ["Connecticut", "Hartford"],
    "Buffalo, NY": ["buffalo"],
    "Birmingham, AL": ["birmingham"]
}



# --- Helper Functions: NEWS (SerpAPI) ---

def get_serp_api_results(api_key, query, num_articles, from_date, to_date, location_query=None):
    """
    Queries the SERP API (Google News) for the given query.
    """
    import requests

    search_url = "https://serpapi.com/search"

    date_filter = f"cd_min:{from_date.strftime('%m/%d/%Y')},cd_max:{to_date.strftime('%m/%d/%Y')}"

    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": num_articles,
        "tbs": f"cdr:1,{date_filter}",
        "tbm": "nws",
        "gl": "us",
        "hl": "en"
    }

    if location_query:
        params["location"] = location_query

    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"SERP API request failed: {e}")
        return None


# --- Helper Functions: SHARED / GEMINI ---

def get_source_name(article):
    source = article.get("source")
    if isinstance(source, dict):
        return source.get("name", "Unknown")
    elif isinstance(source, str):
        return source
    return "Unknown"


def summarize_with_gemini(api_key, articles):
    """
    Summarizes a list of article-like dicts using the Gemini API.
    Each article dict must have at least: title, snippet, source, date, link.
    """
    import google.generativeai as genai
    genai.configure(api_key=api_key)

    prompt_data = [
        {
            "title": a.get("title", "No Title"),
            "snippet": a.get("snippet", "No Snippet"),
            "source": a.get("source", get_source_name(a)),
            "original_date": a.get("date", "Unknown")
        } for a in articles
    ]

    if not prompt_data:
        st.warning("No items found to summarize.")
        return []

    system_prompt = """
    You are an expert incident analyst. Your task is to extract specific information 
    from a list of short text items (news or social media posts).
    
    For EACH item, extract:
    1.  `location`: City, state, or specific address mentioned (e.g., "Houston, TX", "Main St, Springfield"). 
        If no specific location is found, use "Unknown".
    2.  `incident_type`: The type of event (e.g., "House Fire", "Chemical Spill", 
        "Wildfire", "Natural Disaster", "Explosion", "Shooting", "Crash", etc.).
    3.  `incident_date`: The specific date the incident occurred, if mentioned. If not mentioned, use "Unknown".
    4.  `source`: The "source" string for that item (e.g. "X/@user", "Reddit/r/LosAngeles", "NYTimes").
    5.  `summary`: A concise 1-2 sentence summary of the incident described.
    
    Respond ONLY with a valid JSON array, where each object in the array
    corresponds to an input item and contains the fields listed above.
    """

    model = genai.GenerativeModel(
        "gemini-2.5-flash-preview-09-2025",
        system_instruction=system_prompt
    )

    user_prompt = f"""
    Here is a JSON array of items:
    {json.dumps(prompt_data, indent=2)}
    
    Extract the requested fields for each item. 
    Return ONLY a valid JSON array.
    """

    try:
        resp = model.generate_content(
            user_prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        summaries = json.loads(resp.text)

        # add metadata back in
        for i, s in enumerate(summaries):
            s["article_date"] = articles[i].get("date", "Unknown")
            s["article_link"] = articles[i].get("link", "#")
            s["article_title"] = articles[i].get("title", "No Title")

        return summaries

    except Exception as e:
        st.error(f"Gemini summarization failed: {e}")
        return []


def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")


# --- Helper Functions: SOCIAL (Reddit) ---

def get_reddit_incidents_for_metro(metro_area, keywords, lookback_hours=12, limit=50):
    """
    Fetch recent Reddit posts from metro-specific subreddits and filter by incident keywords.
    Uses /new.json (not /search.json) to avoid 403 blocks.
    
    Returns a list of article-like dicts:
        {title, snippet, source, date, link}
    """
    import requests
    from datetime import datetime, timedelta

    subreddits = METRO_REDDIT_MAP.get(metro_area)
    if not subreddits:
        return []

    # Normalize keywords to lowercase for matching
    keywords_lower = [k.lower() for k in keywords]

    headers = {
        "User-Agent": "incident-watcher-bot/0.1 (by u/yourusername)"
    }

    cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
    cutoff_ts = cutoff.timestamp()

    incidents = []

    for sub in subreddits:
        url = f"https://www.reddit.com/r/{sub}/new.json"
        params = {
            "limit": min(limit * 2, 100),  # fetch more, filter down
            "raw_json": 1
        }

        try:
            r = requests.get(url, params=params, headers=headers, timeout=10)
            r.raise_for_status()
        except Exception as e:
            st.warning(f"Reddit request failed for r/{sub}: {e}")
            continue

        data = r.json()
        children = data.get("data", {}).get("children", [])

        for child in children:
            post = child.get("data", {})
            created_utc = post.get("created_utc")
            if created_utc is None:
                continue

            # Time filter
            if created_utc < cutoff_ts:
                continue

            title = post.get("title", "Reddit Post") or ""
            body = post.get("selftext", "") or ""
            text_for_match = (title + " " + body).lower()

            # Keyword filter
            if not any(k in text_for_match for k in keywords_lower):
                continue

            created_dt = datetime.utcfromtimestamp(created_utc).isoformat() + "Z"
            permalink = post.get("permalink", "")

            incidents.append({
                "title": title,
                "snippet": body if body else title,
                "source": f"Reddit/r/{sub}",
                "date": created_dt,
                "link": f"https://www.reddit.com{permalink}"
            })

    # Sort newest first and cap to limit
    def parse_dt(x):
        try:
            return datetime.fromisoformat(x.replace("Z", ""))
        except Exception:
            return datetime.min

    incidents.sort(key=lambda x: parse_dt(x.get("date", "")), reverse=True)
    return incidents[:limit]


# --- Main Streamlit App ---

def main():

    st.set_page_config(
        page_title="Incident Summarizer",
        page_icon="🔥",
        layout="wide"
    )

    # ---- LOAD SECRETS (Streamlit Cloud) ----
    try:
        SERP_API_KEY = st.secrets["SERP_API_KEY"]
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        st.error("Missing SERP_API_KEY or GEMINI_API_KEY in Streamlit secrets.")
        st.stop()

    APP_PASSWORD = st.secrets.get("APP_PASSWORD", None)

    # ---- PASSWORD GATE ----
    if APP_PASSWORD:
        if "authed" not in st.session_state:
            st.session_state.authed = False

        if not st.session_state.authed:
            st.title("🔐 Protected Incident Reporter")

            pw = st.text_input("Enter password:", type="password")

            if pw == APP_PASSWORD:
                st.session_state.authed = True
                st.success("Access granted!")
                st.rerun()
            elif pw:
                st.error("Incorrect password.")
                st.stop()
            else:
                st.stop()

    # ---- SIDEBAR CONFIG ----
    st.sidebar.title("Search Options")
    st.sidebar.success("Secrets loaded successfully")

    # Data source: News vs Social
    source_mode = st.sidebar.radio(
        "Data Source",
        ["News (Google)", "Social (Reddit)"],
        index=0
    )

    # Metro selection (sorted, Nationwide first)
    metros_sorted = ["Nationwide"] + sorted(
        [m for m in TOP_50_US_METROS if m != "Nationwide"]
    )

    metro_area = st.sidebar.selectbox(
        "Metro Area",
        options=metros_sorted
    )

    # Number of items to summarize
    num_items = st.sidebar.slider(
        "Items to Summarize",
        min_value=5,
        max_value=25,
        value=10
    )

    # Date range for NEWS mode
    st.sidebar.subheader("Date Range (News Mode)")
    default_end = datetime.now()
    default_start = default_end - timedelta(days=2)

    col1, col2 = st.sidebar.columns(2)
    from_date = col1.date_input("From", value=default_start)
    to_date = col2.date_input("To", value=default_end)

    if from_date > to_date:
        st.sidebar.error("Invalid date range.")
        return

    # Lookback window for SOCIAL mode
    social_lookback_hours = 12
    if source_mode == "Social (Reddit)":
        social_lookback_hours = st.sidebar.slider(
            "Look back (hours) for Reddit",
            min_value=1,
            max_value=72,
            value=12
        )

    # ---- MAIN UI ----
    st.title("🔥 Local & National Incident Reporter")
    st.write("Summarize incidents from **News** or **social media** using Gemini 2.5 Flash.")

    if st.button("Search for Incidents", type="primary"):

        # Common incident keywords
        incident_keywords = [
            "fire", "explosion", "natural disaster", "chemical spill", "hazmat",
            "building collapse", "power outage", "gas leak", "train derailment",
            "highway closure", "industrial accident", "wildfire", "house fire",
            "apartment fire", "structure fire", "crash", "evacuation","hurricane","flood","tornado","storm","damage"
        ]

        # Decide which source to use
        if source_mode == "News (Google)":
            # --- NEWS MODE ---
            keyword_query = " OR ".join(f'"{k}"' for k in incident_keywords)

            if metro_area == "Nationwide":
                search_query = f"({keyword_query})"
                location_param = None
            else:
                city = metro_area.split(",")[0].strip()
                search_query = f"({keyword_query}) \"{city}\""
                location_param = metro_area

            with st.spinner("Searching Google News..."):
                results = get_serp_api_results(
                    SERP_API_KEY,
                    search_query,
                    100,
                    from_date,
                    to_date,
                    location_query=location_param
                )

            if results and "news_results" in results:
                articles = results["news_results"][:num_items]

                st.success(
                    f"[News] Found {len(results['news_results'])} articles. Summarizing {len(articles)}..."
                )

                with st.spinner("Summarizing with Gemini..."):
                    summaries = summarize_with_gemini(GEMINI_API_KEY, articles)

            else:
                st.error("No news results found.")
                summaries = []

        else:
            # --- SOCIAL MODE (Reddit) ---
            if metro_area == "Nationwide":
                st.warning("For Social mode, please select a specific metro (not Nationwide).")
                summaries = []
            else:
                with st.spinner("Fetching Reddit posts..."):
                    incidents = get_reddit_incidents_for_metro(
                        metro_area,
                        incident_keywords,
                        lookback_hours=social_lookback_hours,
                        limit=num_items
                    )

                if not incidents:
                    st.warning(
                        f"No recent Reddit posts found for {metro_area} "
                        f"matching incident keywords in the last {social_lookback_hours} hours."
                    )
                    summaries = []
                else:
                    st.success(
                        f"[Social] Found {len(incidents)} recent posts. Summarizing with Gemini..."
                    )
                    with st.spinner("Summarizing with Gemini..."):
                        summaries = summarize_with_gemini(GEMINI_API_KEY, incidents)

        # ---- Display results ----
        if summaries:
            import pandas as pd

            st.subheader("Incident Summaries")

            df = pd.DataFrame(summaries)

            cols_order = [
                "article_date", "incident_date", "incident_type",
                "location", "summary", "source", "article_title", "article_link"
            ]
            df = df[[c for c in cols_order if c in df.columns]]

            st.dataframe(df)

            csv = convert_df_to_csv(df)
            st.download_button(
                "Download CSV",
                csv,
                file_name=f"incident_report_{source_mode.replace(' ', '_')}_{metro_area}_{datetime.now().strftime('%Y%m%d')}.csv"
            )
        else:
            st.info("No summaries to display yet.")


if __name__ == "__main__":
    main()


