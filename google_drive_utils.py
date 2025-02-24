import os
import openpyxl





def create_excel_file(file_name, data):
    """Create an Excel file locally."""
    # Create a workbook
    wb = openpyxl.Workbook()
    ws = wb.active
    
    # Add data to worksheet
    for row in data:
        ws.append(row)
    
    # Save the file
    wb.save(file_name)
