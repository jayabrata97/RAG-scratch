import os
import requests

pdf_path = "../source_pdf/human-nutrition-text.pdf"

# Check for the path
if not os.path.exists(pdf_path):
    print("File does not exist, downloading...")
    pdf_url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
    local_filename = pdf_path
    # Send the GET request to the URL
    response = requests.get(pdf_url)
    # Check if the request was successful
    if response.status_code == 200:
        with open(local_filename, "wb") as file:
            file.write(response.content)
        print(f"The PDF has been downloaded and saved as {local_filename}")
    else:
        print(f"Failed to download the PDF with status code: {response.status_code}")
else:
    print(f"File {pdf_path} exists.")
