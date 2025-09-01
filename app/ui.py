import streamlit as st

def upload_pdfs():
    """PDF uploader widget"""
    return st.file_uploader(
        'Select your PDF(s)',
        type='pdf',
        accept_multiple_files=True,
        help="You can upload one or more medical PDF files"
    )
