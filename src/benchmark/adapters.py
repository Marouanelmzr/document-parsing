"""Two input adapters cover all three models:

- vision_chat: a normal chat-formatted VLM (Qwen3-VL, Mistral-Small) that
  gets the schema spelled out as text in the prompt, JSON enforced via
  guided decoding. Used by BOTH Qwen and Mistral -- they don't need
  separate scripts, just the same adapter with different LLM() kwargs.
- nuextract: NuExtract's own template/instructions mechanism. Image-only
  turn; the schema is passed through chat_template_kwargs instead of a
  text prompt.

Keeping the prompt content byte-identical to your original scripts so the
comparison across models isn't confounded by prompt-wording changes.
"""
from pathlib import Path

from common import load_image

# vision_chat family (Qwen3-VL, Mistral-Small)
TS_SCHEMA = """type Address = {
  address: string | null;
  street_number: string | null;
  street_name: string | null;
  po_box: string | null;
  address_complement: string | null; // e.g. floor/building/suite
  city: string | null;
  postal_code: string | null;
  state: string | null;
  country: string | null;
};

type Locale = {
  language: string | null;  // ISO 639-1
  country: string | null;   // ISO 3166-1 alpha-2
  currency: string | null;  // ISO 4217
};

type Tax = {
  rate: number | null;   // decimal, e.g. 0.20
  base: number | null;   // amount tax computed on
  amount: number | null;
};

type LineItem = {
  description: string | null;
  quantity: number | null;
  unit_price: number | null;
  total_price: number | null;  // printed line total
  tax_amount: number | null;
  tax_rate: number | null;     // decimal
  unit_measure: string | null;
};

type Discount = {
  rate: number | null;   // decimal, e.g. 0.0185
  amount: number | null;
};

type Invoice = {
  fields: {
    supplier_name: string | null;
    supplier_phone_number: string | null;
    supplier_address: Address | null;
    customer_name: string | null;
    customer_phone_number: string | null;
    customer_address: Address | null;
    invoice_number: string | null;
    document_type: "invoice" | "tax_invoice" | null;
    date: string | null;      // invoice issue date
    due_date: string | null;  // payment deadline
    period: string | null;    // billing period covered
    locale: Locale | null;
    total_net: number | null;    // total before taxes
    total_tax: number | null;
    total_amount: number | null; // final total the customer owes
    taxes: Tax[];
    discount: Discount | null;
    line_items: LineItem[];
  };
};"""

VISION_CHAT_PROMPT = f"""You are an information-extraction engine for invoice images. Read the attached invoice image and extract every field you can find.

Extract fields matching this shape (null when a field isn't found, [] for empty lists):

{TS_SCHEMA}

CRITICAL RULES:

1. TWO PARTIES ONLY: SUPPLIER (issuer; may be labeled "From"/"Seller"/"Bill From") vs CUSTOMER (buyer; "To"/"Buyer"/"Bill To"/"Ship To"). For address/phone fields, only use values printed under that party's own block. Never cross-copy between supplier_* and customer_*. If unsure which block a value belongs to, use null.

2. NO INFERRED OR COPIED VALUES: a field with no explicit printed label is null -- never fill it by duplicating a different field's value.

3. LINE ITEM COLUMNS ARE POSITIONAL: map each value by its column position in the table header (e.g. description, quantity, unit price, tax rate, tax amount, total price) -- never reuse one printed number for two fields. tax_amount and total_price are almost always different; only set them equal if the table genuinely prints the same number in both columns.

4. Transcribe values exactly as printed (dates, numbers, names, addresses); do not compute or validate totals. Split a combined "bill to"/buyer block into customer_name and customer_address. Fill line_items with one entry per row, even if some sub-fields are missing.

5. THREE DISTINCT DATES -- do not confuse them:
   - date = invoice issue date. Keywords: "Date", "Date de facture", "Facturé le", "Invoice date", "Date of issue". Usually the earliest date, near the invoice number. Never the payment deadline.
   - due_date = payment deadline. Keywords (FR): "Échéance", "Date d'échéance", "À payer avant le", "Payable au", "Date limite de paiement", "Valable jusqu'au". (EN): "Due date", "Payment due", "Payable by", "Deadline". Usually >= the issue date -- if two dates appear, the later one is usually due_date.
   - period = the timeframe the invoiced work covers (not issue date, not deadline). Keywords: "Période", "Prestations du", "Mois de", "Billing period". Often a month or date range; null if the invoice is a one-off with no stated period.

6. document_type must be exactly one of: invoice, tax_invoice.
7. NEVER CALCULATE, ONLY TRANSCRIBE: total_net, total_tax, total_amount, and discount.amount/discount.rate must each be copied from a number that is explicitly printed on the invoice under that meaning. Do not derive any of them by summing line items, subtracting a discount, multiplying a rate by a base, or any other arithmetic -- even if the computed value would be "more correct" than what's printed. If a given total or the discount isn't printed anywhere on the document, its value is null.
"""


def build_vision_chat_conversation(img_path: Path, size_log: list):
    image = load_image(img_path, size_log)
    return [{
        "role": "user",
        "content": [
            {"type": "image_pil", "image_pil": image},
            {"type": "text", "text": VISION_CHAT_PROMPT},
        ],
    }]


# nuextract family
NUEXTRACT_TEMPLATE = {
    "supplier_name": "verbatim-string",
    "supplier_phone_number": "verbatim-string",
    "supplier_address": {
        "address": "verbatim-string", "street_number": "verbatim-string",
        "street_name": "verbatim-string", "po_box": "verbatim-string",
        "address_complement": "verbatim-string", "city": "verbatim-string",
        "postal_code": "verbatim-string", "state": "verbatim-string", "country": "country",
    },
    "customer_name": "verbatim-string",
    "customer_address": {
        "address": "verbatim-string", "street_number": "verbatim-string",
        "street_name": "verbatim-string", "po_box": "verbatim-string",
        "address_complement": "verbatim-string", "city": "verbatim-string",
        "postal_code": "verbatim-string", "state": "verbatim-string", "country": "country",
    },
    "invoice_number": "verbatim-string",
    "document_type": ["invoice", "tax_invoice"],
    "date": "date-time",
    "due_date": "date-time",
    "period": "verbatim-string",
    "locale": {"language": "verbatim-string", "country": "country", "currency": "currency"},
    "total_net": "number",
    "total_tax": "number",
    "total_amount": "number",
    "taxes": [{"rate": "number", "base": "number", "amount": "number"}],
    "line_items": [{
        "description": "verbatim-string", "quantity": "number", "unit_price": "number",
        "total_price": "number", "tax_amount": "number", "tax_rate": "number",
        "unit_measure": "verbatim-string",
    }],
}

NUEXTRACT_INSTRUCTIONS = """TWO PARTIES ONLY: SUPPLIER (issuer; may be labeled "From"/"Seller"/"Bill From") vs CUSTOMER \
(buyer; "To"/"Buyer"/"Bill To"/"Ship To"). For address/phone fields, only use values printed under that \
party's own block. Never cross-copy between supplier_* and customer_*. If unsure which block a value \
belongs to, leave it null.

NO INFERRED OR COPIED VALUES: a field with no explicit printed label is null -- never fill it by \
duplicating a different field's value (e.g. do not copy invoice_number into a missing PO field).

LINE ITEM COLUMNS ARE POSITIONAL: map each value by its column position in the table header (description, \
quantity, unit price, tax rate, tax amount, total price) -- never reuse one printed number for two fields. \
tax_amount and total_price are almost always different; only set them equal if the table genuinely prints \
the same number in both columns.

Transcribe values exactly as printed; do not compute or validate totals. Split a combined "bill to"/buyer \
block into customer_name and customer_address. Fill line_items with one entry per row, even if some \
sub-fields are missing.

THREE DISTINCT DATES:
- date = invoice issue date. Keywords: "Date", "Date de facture", "Facturé le", "Invoice date", "Date of \
issue". Usually the earliest date, near the invoice number. Never the payment deadline.
- due_date = payment deadline. Keywords (FR): "Échéance", "Date d'échéance", "À payer avant le", "Payable \
au", "Date limite de paiement", "Valable jusqu'au". (EN): "Due date", "Payment due", "Payable by", \
"Deadline". Usually on or after the issue date -- if two dates appear, the later one is usually due_date.
- period = the timeframe the invoiced work covers (not issue date, not deadline). Keywords: "Période", \
"Prestations du", "Mois de", "Billing period". Often a month or date range; null if the invoice is \
a one-off with no stated period.
"""


def build_nuextract_conversation(img_path: Path, size_log: list):
    image = load_image(img_path, size_log)
    return [{"role": "user", "content": [{"type": "image_pil", "image_pil": image}]}]