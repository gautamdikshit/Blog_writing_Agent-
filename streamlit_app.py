import streamlit as st
import os
from datetime import datetime
from pathlib import Path
from agent import run_blog_agent

BLOG_DIR = Path("blogs")
BLOG_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="AI Blog Generator", layout="wide")

st.title("📝 AI Technical Blog Generator")

# --------------------------
# Sidebar – Recent Blogs
# --------------------------
st.sidebar.title("📚 Recent Blogs")

blog_files = sorted(BLOG_DIR.glob("*.md"), reverse=True)

selected_blog = None

for blog_file in blog_files:
    if st.sidebar.button(blog_file.stem):
        selected_blog = blog_file

# --------------------------
# Main Input
# --------------------------
topic = st.text_input("Enter a blog topic")

if st.button("Generate"):
    if not topic.strip():
        st.warning("Please enter a topic.")
    else:
        with st.spinner("Generating blog..."):
            blog = run_blog_agent(topic)

        # Save blog to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = topic.replace(" ", "_")[:50]
        filename = BLOG_DIR / f"{timestamp}_{safe_title}.md"
        filename.write_text(blog, encoding="utf-8")

        st.success("Blog generated and saved!")
        st.markdown("---")
        st.markdown(blog)

# --------------------------
# Load Selected Blog
# --------------------------
if selected_blog:
    st.markdown("---")
    st.subheader(f"📖 {selected_blog.stem}")
    content = selected_blog.read_text(encoding="utf-8")
    st.markdown(content)
