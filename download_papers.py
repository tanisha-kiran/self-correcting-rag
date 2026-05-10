#!/usr/bin/env python3
"""
Download research papers from arXiv for RAG system testing
Perfect for technical/software domain papers on AI, ML, and systems
"""

import os
import requests
import time
from pathlib import Path

# Recommended papers for RAG testing
PAPERS = [
    {
        "title": "Retrieval-Augmented Generation for Large Language Models: A Survey",
        "arxiv_id": "2312.10997",
        "filename": "01_RAG_Survey.pdf"
    },
    {
        "title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        "arxiv_id": "2310.11511",
        "filename": "02_Self_RAG.pdf"
    },
    {
        "title": "Corrective Retrieval Augmented Generation",
        "arxiv_id": "2401.15884",
        "filename": "03_CRAG.pdf"
    },
    {
        "title": "LangChain: Building Applications with LLMs through Composability",
        "arxiv_id": "2311.06151",
        "filename": "04_LangChain.pdf"
    },
    {
        "title": "Agents as a New Paradigm for Software Engineering",
        "arxiv_id": "2401.01997",
        "filename": "05_Agent_Systems.pdf"
    },
    {
        "title": "The Unreasonable Effectiveness of Deep Learning in Artificial Intelligence",
        "arxiv_id": "1906.00004",
        "filename": "06_Deep_Learning_Effectiveness.pdf"
    }
]

def download_paper(arxiv_id: str, filename: str, docs_dir: str = "documents") -> bool:
    """Download a paper from arXiv"""
    
    # Create documents directory if it doesn't exist
    Path(docs_dir).mkdir(exist_ok=True)
    
    filepath = os.path.join(docs_dir, filename)
    
    # Skip if already exists
    if os.path.exists(filepath):
        print(f"  ✓ {filename} already exists")
        return True
    
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    try:
        print(f"  ⬇️  Downloading {filename}...", end=" ")
        response = requests.get(pdf_url, timeout=30)
        
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            print(f"✓ ({file_size:.1f} MB)")
            return True
        else:
            print(f"✗ HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Download all recommended papers"""
    
    print("=" * 70)
    print("Research Paper Downloader for RAG System")
    print("=" * 70)
    print()
    print("📚 Papers to Download:")
    print()
    
    for i, paper in enumerate(PAPERS, 1):
        print(f"{i}. {paper['title']}")
        print(f"   arXiv: {paper['arxiv_id']}")
    
    print()
    print("=" * 70)
    print("Starting download...")
    print("=" * 70)
    print()
    
    successful = 0
    failed = 0
    
    for paper in PAPERS:
        if download_paper(paper["arxiv_id"], paper["filename"]):
            successful += 1
        else:
            failed += 1
        
        # Be nice to arXiv servers
        time.sleep(2)
    
    print()
    print("=" * 70)
    print(f"Download Complete: {successful} successful, {failed} failed")
    print("=" * 70)
    print()
    
    # Check what we have
    docs_dir = "documents"
    if os.path.exists(docs_dir):
        pdf_count = len([f for f in os.listdir(docs_dir) if f.endswith('.pdf')])
        total_size = sum(os.path.getsize(os.path.join(docs_dir, f)) 
                        for f in os.listdir(docs_dir) 
                        if f.endswith('.pdf')) / (1024 * 1024)
        
        print(f"📊 Current Status:")
        print(f"   PDFs in documents/: {pdf_count}")
        print(f"   Total size: {total_size:.1f} MB")
    
    print()
    print("✅ Ready to test the RAG system!")
    print()
    print("Next steps:")
    print("  1. python3 rag_system.py        # Test CRAG")
    print("  2. python3 rag_selfrag.py       # Test Self-RAG")
    print()


if __name__ == "__main__":
    # Check if requests library is available
    try:
        import requests
    except ImportError:
        print("❌ Error: 'requests' library not found")
        print("Install it with: pip install requests")
        exit(1)
    
    main()