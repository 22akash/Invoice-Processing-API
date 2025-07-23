import os
import requests

response = requests.post(
    "https://llmfoundry.straive.com/cluster",
    headers={"Authorization": f"Bearer {'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImFrYXNoLmd1bmR1QGdyYW1lbmVyLmNvbSJ9.gY4pT7cZoafaS2EC3lgLokeRUyvdD99yW6O-Ggydi-A'}:my-test-project"},
    json={"docs": ["Apple", "Orange", "42"], "n": 2}
)
print(response.json())

# import os
# import openai
# from PIL import Image
# import base64
# import io

# # === CONFIGURATION ===
# OPENAI_API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImFrYXNoLmd1bmR1QGdyYW1lbmVyLmNvbSJ9.gY4pT7cZoafaS2EC3lgLokeRUyvdD99yW6O-Ggydi-A'
# BASE_URL = "https://llmfoundry.straive.com/openai/v1/"

# # Set up OpenAI client with custom base URL
# openai.base_url = BASE_URL
# openai.api_key = OPENAI_API_KEY

# # === HELPER: Encode Image to Base64 ===
# def encode_image_to_base64(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode('utf-8')

# # === PROMPT TEMPLATE ===
# def build_invoice_extraction_prompt():
#     return """
# You are an intelligent document processing assistant. The user will provide you with an image of an invoice.

# Extract the following fields accurately:
# - Vendor Name
# - Invoice Number
# - Invoice Date
# - Due Date (if available)
# - Billing Address
# - Shipping Address (if available)
# - Line Items (as a list with Description, Quantity, Unit Price, Total Price)
# - Subtotal
# - Taxes
# - Total Amount
# - Currency

# Respond strictly in the following JSON format:

# {
#     "vendor_name": "...",
#     "invoice_number": "...",
#     "invoice_date": "...",
#     "due_date": "...",
#     "billing_address": "...",
#     "shipping_address": "...",
#     "line_items": [
#         {
#             "description": "...",
#             "quantity": "...",
#             "unit_price": "...",
#             "total_price": "..."
#         }
#     ],
#     "subtotal": "...",
#     "taxes": "...",
#     "total_amount": "...",
#     "currency": "..."
# }

# If any field is missing in the invoice, set its value to null.
# Now extract the data:
# """

# # === MAIN FUNCTION ===
# def extract_invoice_data_with_gpt4o(image_path):
#     base64_image = encode_image_to_base64(image_path)

#     prompt = build_invoice_extraction_prompt()

#     response = openai.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "user", "content": [
#                 {"type": "text", "text": prompt},
#                 {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
#             ]}
#         ],
#         temperature=0.2,
#         max_tokens=1500,
#     )

#     extracted_data = response.choices[0].message.content
#     return extracted_data


# # === RUN ===
# if __name__ == "__main__":
#     invoice_image_path = "data/3.jpg"  # <-- Replace this
#     result = extract_invoice_data_with_gpt4o(invoice_image_path)
#     print(result)
