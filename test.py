import os
import pdfplumber
import pandas as pd
from PyPDF2 import PdfReader # You can keep this if you need it for other tasks
# from utils.logger import setup_logger

def extract_table_from_pdf(pdf_path):
    # 1. Open the PDF with pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        # 2. Select the page you want (e.g., the first page)
        page = pdf.pages[1]
        
        # 3. Extract the table
        # This returns a list of lists, where each inner list is a row
        table = page.extract_table()
        
        if not table:
            print("No table found on this page.")
            return None
            
        # 4. Convert to a Pandas DataFrame for easy manipulation
        # Assuming the first row contains your headers
        headers = table[0]
        data = table[1:]
        
        df = pd.DataFrame(data, columns=headers)
        
        return df

# --- Usage ---
pdf_file = "sample_excel.pdf"

if os.path.exists(pdf_file):
    df = extract_table_from_pdf(pdf_file)
    
    if df is not None:
        print("Successfully extracted table:")
        print(df)
        
        # 5. Save it as an actual Excel file!
        excel_output = "extracted_data2.xlsx"
        df.to_excel(excel_output, index=False)
        print(f"Saved to {excel_output}")
else:
    print(f"File {pdf_file} not found.")