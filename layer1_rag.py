# FinSight Layer 1 — RAG Foundation
# This file grows every week until Week 8
# when it becomes the full production system

from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

from langchain_community.vectorstores import Chroma

from langchain_core.documents import Document

import os

print("="*55)
print("FinSight Layer 1 — RAG Foundation")
print("="*55)

# ── Step 1: Create Financial Documents ────────
# Using sample text now.
# Week 3: replace with real SEC filings
# Week 4: add live stock data
# Week 5: add multi-agent system

print("\nStep 1 — Loading financial documents...")

financial_docs = [
    Document(
        page_content="""
        Apple Inc — Annual Report FY2023
        Total net sales: $394.3 billion
        iPhone revenue: $200.6 billion (51% of total)
        Services revenue: $85.2 billion (grew 9% YoY)
        Mac revenue: $29.4 billion
        Wearables revenue: $39.8 billion
        Gross margin: 44.1%
        Net income: $97.0 billion
        Employees: 161,000 worldwide
        CEO: Tim Cook
        Key risks: Supply chain, geopolitical tensions,
        competition from Samsung and Google,
        regulatory pressure in EU and US.
        """,
        metadata={
            "company"   : "Apple",
            "ticker"    : "AAPL",
            "doc_type"  : "annual_report",
            "year"      : "2023",
            "source"    : "SEC EDGAR"
        }
    ),
    Document(
        page_content="""
        Nvidia Corporation — Annual Report FY2023
        Total revenue: $26.97 billion (up 16% YoY)
        Data Center revenue: $15.0 billion (grew 41%)
        Gaming revenue: $9.07 billion (fell 27%)
        Professional Visualization: $1.5 billion
        Automotive revenue: $903 million
        Gross margin: 56.9%
        Net income: $4.37 billion
        Employees: 26,196 worldwide
        CEO: Jensen Huang
        The surge in AI demand for H100 GPUs drove
        Data Center growth significantly.
        Key risks: Export controls to China,
        competition from AMD and Intel,
        supply constraints from TSMC.
        """,
        metadata={
            "company"   : "Nvidia",
            "ticker"    : "NVDA",
            "doc_type"  : "annual_report",
            "year"      : "2023",
            "source"    : "SEC EDGAR"
        }
    ),
    Document(
        page_content="""
        Microsoft Corporation — Annual Report FY2023
        Total revenue: $211.9 billion (up 7% YoY)
        Intelligent Cloud: $87.9 billion (grew 19%)
        Azure cloud growth: 26% year over year
        Productivity and Business: $69.3 billion
        More Personal Computing: $54.7 billion
        Gross margin: 69.0%
        Net income: $72.4 billion
        Employees: 221,000 worldwide
        CEO: Satya Nadella
        Investment in OpenAI and Copilot integration
        across all products was a major highlight.
        Key risks: Regulatory scrutiny of Activision
        acquisition, competition from AWS and Google,
        cybersecurity threats.
        """,
        metadata={
            "company"   : "Microsoft",
            "ticker"    : "MSFT",
            "doc_type"  : "annual_report",
            "year"      : "2023",
            "source"    : "SEC EDGAR"
        }
    ),
    Document(
        page_content="""
        Infosys Limited — Annual Report FY2023
        Total revenue: $18.21 billion (grew 17% YoY)
        Digital revenue: 62.2% of total revenue
        Operating margin: 21.0%
        Net income: $2.98 billion
        Employees: 343,234 worldwide
        CEO: Salil Parekh
        Headquartered in Bengaluru, India.
        Major clients in BFSI, Retail, Manufacturing.
        Key services: IT consulting, outsourcing,
        cloud transformation, AI/ML services.
        Key risks: Currency fluctuation, visa issues
        for US operations, client concentration risk.
        """,
        metadata={
            "company"   : "Infosys",
            "ticker"    : "INFY",
            "doc_type"  : "annual_report",
            "year"      : "2023",
            "source"    : "Annual Report"
        }
    ),
]

print(f"Loaded {len(financial_docs)} company documents")
for doc in financial_docs:
    print(f"  → {doc.metadata['company']} "
          f"({doc.metadata['ticker']})")
    # ── Step 2: Split into Chunks ──────────────────
print("\nStep 2 — Splitting into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)

chunks = splitter.split_documents(financial_docs)
print(f"Created {len(chunks)} chunks")
print(f"\nSample chunk:")
print(f"Content: {chunks[0].page_content[:150]}...")
print(f"Metadata: {chunks[0].metadata}")
# ── Step 3: Create Embeddings ─────────────────
print("\nStep 3 — Creating embeddings...")
print("(First run downloads model — 2 mins)")

# Free model — runs on your CPU
# No API key needed
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Test embedding
test_embed = embeddings.embed_query("Apple revenue")
print(f"Embedding dimension: {len(test_embed)}")
print(f"First 5 values: "
      f"{[round(v,3) for v in test_embed[:5]]}")
print("Embeddings working!")
# ── Step 4: Store in ChromaDB ─────────────────
print("\nStep 4 — Storing in ChromaDB...")

# Create/load vector store
persist_dir = "./finsight_db"

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_dir
)

print(f"Stored {len(chunks)} chunks in ChromaDB")
print(f"Database saved at: {persist_dir}/")
# ── Step 5: Test Retrieval ─────────────────────
print("\nStep 5 — Testing retrieval...")
print("="*45)

test_queries = [
    "What was Apple's revenue?",
    "What are Nvidia's main risks?",
    "How many employees does Microsoft have?",
    "What is Infosys revenue growth?",
    "Which company has highest gross margin?"
]

for query in test_queries:
    print(f"\nQuery: '{query}'")
    results = vectorstore.similarity_search(
        query, k=2
    )
    print(f"Top result from: "
          f"{results[0].metadata['company']}")
    print(f"Content: "
          f"{results[0].page_content[:100]}...")
    # ── Step 6: Add LLM (optional — needs API key) ─
print("\n" + "="*45)
print("Step 6 — Adding LLM for Q&A...")

# Get FREE API key: groq.com → sign up → API Keys
# Then run: set GROQ_API_KEY=your-key (Windows)
# Or: export GROQ_API_KEY=your-key (Mac/Linux)

groq_key = os.environ.get("GROQ_API_KEY", "")

if groq_key:
    from langchain_groq import ChatGroq
    from langchain_classic.chains import RetrievalQA

    llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_key
     )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(k=3),
        return_source_documents=True
    )

    questions = [
        "Compare Apple and Microsoft revenue",
        "Which company has the most employees?",
        "What are the main risks for Nvidia?"
    ]

    print("\nLLM-powered Q&A:")
    for q in questions:
        result = qa.invoke({"query": q})
        print(f"\nQ: {q}")
        print(f"A: {result['result'][:200]}...")

else:
    print("\nSkipping LLM — add GROQ_API_KEY to use")
    print("Get free key at: groq.com")
    print("Retrieval (Steps 1-5) works without key!")

print("\n" + "="*55)
print("FinSight Layer 1 — RAG Foundation COMPLETE")
print("="*55)
print("\nWhat we have:")
print("  ✅ 4 company financial documents loaded")
print("  ✅ Documents split into chunks")
print("  ✅ BGE embeddings created (free, local)")
print("  ✅ ChromaDB vector store built + saved")
print("  ✅ Semantic search working")
print("  ✅ Optional LLM Q&A layer")
print("\nNext week:")
print("  → Real SEC filings from EDGAR API")
print("  → Hybrid search (BM25 + vector)")
print("  → RAGAS evaluation")
print("  → Agent with tools")