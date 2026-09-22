from __future__ import annotations

import json

import re

from pathlib import Path

from typing import Any


# ============================================================
# Configuration
# ============================================================

DIRECT_FIELD_MAP = {
    "SELLER_NAME": "supplier_name",
    "NUMBER": "invoice_number",
    "DATE": "date",
    "DUE_DATE": "due_date",
}

NUMERIC_FIELD_MAP = {
    "SUB_TOTAL": "total_net",
    "SUBTOTAL": "total_net",
    "TOTAL_TAX": "total_tax",
    "TOTAL TAX": "total_tax",
    "TOTAL": "total_amount",
}

ADDRESS_KEYS = [
    "address",
    "street_number",
    "street_name",
    "po_box",
    "address_complement",
    "city",
    "postal_code",
    "state",
    "country",
]


# ============================================================
# Generic helpers
# ============================================================

def clean_text(text: Any) -> str:
    """Normalize whitespace while preserving meaningful punctuation."""
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def get_text(value: Any) -> str | None:
    """
    Extract the text value from a FATURA field.

    FATURA fields normally look like:
        {"bbox": [...], "text": "..."}
    """
    if value is None:
        return None
    if isinstance(value, dict):
        text = value.get("text")
        if text is not None:
            text = clean_text(text)
            return text or None
    elif isinstance(value, str):
        text = clean_text(value)
        return text or None
    return None


def get_first_text(
    annotation: dict[str, Any],
    *field_names: str,
) -> str | None:
    """Return the first non-empty text among the requested FATURA fields."""
    for field_name in field_names:
        text = get_text(annotation.get(field_name))
        if text:
            return text
    return None


def parse_float(value: Any) -> float | None:
    """Parse a numeric value robustly."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(" ", "")
    # Example:
    # 1,234.56 -> 1234.56
    if "," in text and "." in text:
        text = text.replace(",", "")
    # Example:
    # 1234,56 -> 1234.56
    elif "," in text:
        text = text.replace(",", ".")
    match = re.search(
        r"[-+]?\d+(?:\.\d+)?",
        text,
    )
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


# ============================================================
# Date parsing
# ============================================================

def parse_date_text(text: str | None) -> str | None:
    """
    Extract the invoice date while preserving the original
    formatting.

    Example:
        Invoice Date: 11-May-2013
        -> 11-May-2013
    """
    if not text:
        return None
    text = clean_text(text)
    match = re.search(
        r"(?:Invoice\s+Date|Date)\s*:\s*(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_due_date_text(text: str | None) -> str | None:
    """
    Extract the due date while preserving the original
    formatting.

    Example:
        Due Date : 21-May-2016
        -> 21-May-2016
    """
    if not text:
        return None
    text = clean_text(text)
    match = re.search(
        r"Due\s+Date\s*:?\s*(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return text.strip()


# ============================================================
# Invoice number
# ============================================================

def parse_invoice_number_text(
    text: str | None,
) -> str | None:
    """
    Extract invoice number from fields such as:

        Invoice Number: 123
        Invoice No: 123
        Invoice #: 123
        Invoice ID: 123
        PO Number : 99
    """
    if not text:
        return None
    text = clean_text(text)
    text = re.sub(
        r"^\s*"
        r"(?:"
        r"Invoice\s+(?:Number|No\.?|#|ID)"
        r"|PO\s+Number"
        r")"
        r"\s*:?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip() or None


# ============================================================
# Address parsing
# ============================================================

def empty_address() -> dict[str, Any]:
    """Return the complete target address schema with null values."""
    return {
        key: None
        for key in ADDRESS_KEYS
    }


def normalize_address_schema(
    address: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Ensure every address contains the complete target schema.
    Missing fields are explicitly represented as null.
    """
    if not address:
        return empty_address()
    return {
        key: address.get(key)
        for key in ADDRESS_KEYS
    }


def extract_address_complement(
    street: str,
) -> tuple[str, str | None]:
    """
    Extract common address complements from a street string.

    Examples:
        "Woods Drive Apt. 239"
            -> ("Woods Drive", "Apt. 239")

        "Michelle Mall Suite 662"
            -> ("Michelle Mall", "Suite 662")

        "Dunn Ferry Apt. 021"
            -> ("Dunn Ferry", "Apt. 021")
    """
    if not street:
        return street, None
    complement_pattern = re.compile(
        r"\s+("
        r"Apt\.?\s+[A-Za-z0-9\-]+"
        r"|Apartment\s+[A-Za-z0-9\-]+"
        r"|Suite\s+[A-Za-z0-9\-]+"
        r"|Ste\.?\s+[A-Za-z0-9\-]+"
        r"|Unit\s+[A-Za-z0-9\-]+"
        r"|Floor\s+[A-Za-z0-9\-]+"
        r"|Fl\.?\s+[A-Za-z0-9\-]+"
        r"|Building\s+[A-Za-z0-9\-]+"
        r"|Bldg\.?\s+[A-Za-z0-9\-]+"
        r")"
        r"\s*$",
        flags=re.IGNORECASE,
    )
    match = complement_pattern.search(street)
    if not match:
        return street.strip(), None
    complement = match.group(1).strip()
    street_without_complement = street[:match.start()].strip()
    return street_without_complement, complement


def parse_address_text(
    text: str | None,
) -> dict[str, Any] | None:
    """
    Parse a relatively structured FATURA address.

    Example:
        Address:16424 Timothy Mission
        Markville, AK 58294 US

    becomes:
        {
            "address": "16424 Timothy Mission",
            "street_number": "16424",
            "street_name": "Timothy Mission",
            "po_box": null,
            "address_complement": null,
            "city": "Markville",
            "postal_code": "58294",
            "state": "AK",
            "country": "US"
        }
    """
    if not text:
        return None
    text = clean_text(text)
    # Remove FATURA label.
    text = re.sub(
        r"^\s*Address\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    if not text:
        return None
    lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]
    if not lines:
        return None
    address = empty_address()
    # --------------------------------------------------------
    # PO Box
    # --------------------------------------------------------
    full_text = " ".join(lines)
    po_match = re.search(
        r"\bP\.?\s*O\.?\s*Box\s+([A-Za-z0-9\-]+)",
        full_text,
        flags=re.IGNORECASE,
    )
    if po_match:
        address["po_box"] = po_match.group(1)
    # --------------------------------------------------------
    # Location line
    #
    # Example:
    # Markville, AK 58294 US
    # --------------------------------------------------------
    location_index = None
    location_match = None
    location_pattern = re.compile(
        r"^(?P<city>.+?),\s*"
        r"(?P<state>[A-Za-z]{2})\s+"
        r"(?P<postal>[A-Za-z0-9\- ]+?)\s+"
        r"(?P<country>[A-Za-z]{2})$"
    )
    for i, line in enumerate(lines):
        match = location_pattern.match(line)
        if match:
            location_index = i
            location_match = match
            break
    if location_match:
        address["city"] = (
            location_match.group("city").strip()
        )
        address["state"] = (
            location_match.group("state").strip()
        )
        address["postal_code"] = (
            location_match.group("postal").strip()
        )
        address["country"] = (
            location_match.group("country").strip()
        )
        street_lines = lines[:location_index]
    else:
        street_lines = lines
    # --------------------------------------------------------
    # Street address
    # --------------------------------------------------------
    if street_lines:
        street = " ".join(street_lines).strip()
        # Remove PO Box from street if detected.
        if address["po_box"]:
            street = re.sub(
                r"\bP\.?\s*O\.?\s*Box\s+"
                + re.escape(str(address["po_box"])),
                "",
                street,
                flags=re.IGNORECASE,
            ).strip()
        street = re.sub(
            r"\s+",
            " ",
            street,
        ).strip()
        if street:
            # ------------------------------------------------
            # Address complement
            # ------------------------------------------------
            street, complement = extract_address_complement(
                street
            )
            if complement:
                address["address_complement"] = complement
            # ------------------------------------------------
            # Full address
            # ------------------------------------------------
            address["address"] = street
            # ------------------------------------------------
            # Street number + street name
            # ------------------------------------------------
            street_match = re.match(
                r"^(?P<number>\d+[A-Za-z\-]*)\s+"
                r"(?P<name>.+)$",
                street,
            )
            if street_match:
                address["street_number"] = (
                    street_match.group("number").strip()
                )
                address["street_name"] = (
                    street_match.group("name").strip()
                )
            elif not address["po_box"]:
                address["street_name"] = street
    return normalize_address_schema(address)


# ============================================================
# Customer / supplier extraction
# ============================================================


def parse_buyer_text(
    text: str | None,
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """
    Parse BUYER field.

    Handles both:
        Buyer:Angela Wilson

    and:
        Bill to:Crystal Beck

    Also extracts customer phone numbers from lines such as:
        Tel: +1 555 123 4567
        Phone: +1 555 123 4567
        Telephone: +1 555 123 4567
        Mobile: +1 555 123 4567
        Phone Number: +1 555 123 4567
    """
    if not text:
        return None, None, None

    text = clean_text(text)

    lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    if not lines:
        return None, None, None

    # --------------------------------------------------------
    # Customer name
    # --------------------------------------------------------

    first_line = lines[0]
    address_start = 1

    buyer_match = re.search(
        r"\bBuyer\s*:\s*(.+)$",
        first_line,
        flags=re.IGNORECASE,
    )

    bill_to_inline_match = re.search(
        r"\bBill\s*[\-_ ]?\s*to\s*:\s*(.+)$",
        first_line,
        flags=re.IGNORECASE,
    )

    bill_to_label_match = re.fullmatch(
        r"Bill\s*[\-_ ]?\s*to\s*:?",
        first_line,
        flags=re.IGNORECASE,
    )

    ship_to_label_match = re.fullmatch(
        r"Ship\s*[\-_ ]?\s*to\s*:?",
        first_line,
        flags=re.IGNORECASE,
    )

    if buyer_match:
        customer_name = buyer_match.group(1).strip()

    elif bill_to_inline_match:
        customer_name = bill_to_inline_match.group(1).strip()

    elif bill_to_label_match:
        if len(lines) > 1:
            customer_name = lines[1].strip()
            address_start = 2
        else:
            customer_name = None

    elif ship_to_label_match:
        if len(lines) > 1:
            customer_name = lines[1].strip()
            address_start = 2
        else:
            customer_name = None

    else:
        customer_name = first_line.strip()

    # --------------------------------------------------------
    # Customer phone
    # --------------------------------------------------------

    customer_phone_number = None

    phone_pattern = re.compile(
        r"^(?:Tel|Telephone|Phone|Phone\s*Number|Mobile|Mobile\s*Number)"
        r"\s*:\s*(.+)$",
        flags=re.IGNORECASE,
    )

    # --------------------------------------------------------
    # Address lines
    # --------------------------------------------------------

    address_lines = []

    for line in lines[address_start:]:

        phone_match = phone_pattern.match(line)

        if phone_match:
            if customer_phone_number is None:
                customer_phone_number = (
                    phone_match.group(1).strip()
                )
            continue

        if re.match(
            r"^(?:Email|Site|Website)\s*:",
            line,
            flags=re.IGNORECASE,
        ):
            continue

        if re.match(
            r"^GSTIN\s*:",
            line,
            flags=re.IGNORECASE,
        ):
            continue

        address_lines.append(line)

    # --------------------------------------------------------
    # Customer address
    # --------------------------------------------------------

    customer_address = None

    if address_lines:
        customer_address = parse_address_text(
            "\n".join(address_lines)
        )

    return (
        customer_name or None,
        customer_address,
        customer_phone_number,
    )


def parse_seller_address_text(
    text: str | None,
) -> dict[str, Any] | None:
    """Parse the SELLER_ADDRESS field."""
    return parse_address_text(text)


# ============================================================
# Numeric fields
# ============================================================

def extract_numeric_field(
    text: str | None,
) -> float | None:
    """
    Extract a numeric amount from a simple FATURA numeric field.

    Example:
        SUB_TOTAL : 311.26 USD
        -> 311.26
    """
    if not text:
        return None
    text = clean_text(text)
    matches = re.findall(
        r"[-+]?\d[\d,]*(?:[.,]\d+)?",
        text,
    )
    if not matches:
        return None
    return parse_float(matches[0])


def extract_tax_rate(
    text: str | None,
) -> float | None:
    """
    Extract tax rate as a decimal fraction.

    Example:
        TAX:VAT (6.4%): 13.03 USD
        -> 0.064
    """
    if not text:
        return None

    text = clean_text(text)

    match = re.search(
        r"\(\s*"
        r"([-+]?\d[\d,]*(?:[.,]\d+)?)"
        r"\s*%\s*\)",
        text,
    )

    if match:
        rate = parse_float(match.group(1))
        return rate / 100 if rate is not None else None

    match = re.search(
        r"([-+]?\d[\d,]*(?:[.,]\d+)?)\s*%",
        text,
    )

    if match:
        rate = parse_float(match.group(1))
        return rate / 100 if rate is not None else None

    return None

def extract_tax_amount(
    text: str | None,
) -> float | None:
    """
    Extract tax amount.

    Example:
        TAX:VAT (4.18%): 13.03 USD

    returns:
        13.03
    """
    if not text:
        return None
    text = clean_text(text)
    # Main FATURA pattern:
    #
    # (4.18%): 13.03
    #
    match = re.search(
        r"%\s*\)\s*:\s*"
        r"([-+]?\d[\d,]*(?:[.,]\d+)?)",
        text,
    )
    if match:
        return parse_float(match.group(1))
    # Fallback:
    # use the last numeric value in the TAX field.
    numbers = re.findall(
        r"[-+]?\d[\d,]*(?:[.,]\d+)?",
        text,
    )
    if numbers:
        return parse_float(numbers[-1])
    return None


# ============================================================
# GST tax extraction
# ============================================================

def extract_gst_tax(
    text: str | None,
) -> dict[str, float] | None:
    """
    Extract GST rate and amount.

    Example:
        GST(6.4%) : 13.03
        -> {"rate": 0.064, "amount": 13.03}
    """
    if not text:
        return None

    text = clean_text(text)

    match = re.search(
        r"GST\s*\(\s*"
        r"([-+]?\d[\d,]*(?:[.,]\d+)?)"
        r"\s*%\s*\)\s*:\s*"
        r"([-+]?\d[\d,]*(?:[.,]\d+)?)",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    rate = parse_float(match.group(1))
    amount = parse_float(match.group(2))

    if rate is None or amount is None:
        return None

    return {
        "rate": rate / 100,
        "amount": amount,
    }

# ============================================================
# Discount extraction
# ============================================================ 

def extract_discount(
    text: str | None,
) -> dict[str, float] | None:
    """
    Extract discount rate and amount.

    Example:
        DISCOUNT(1.85%): (-) 13.42
        -> {"rate": 0.0185, "amount": 13.42}
    """
    if not text:
        return None

    text = clean_text(text)

    match = re.search(
        r"DISCOUNT\s*\(\s*"
        r"([-+]?\d[\d,]*(?:[.,]\d+)?)"
        r"\s*%\s*\)\s*:\s*"
        r"(?:\(\s*-\s*\)\s*)?"
        r"([-+]?\d[\d,]*(?:[.,]\d+)?)",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    rate = parse_float(match.group(1))
    amount = parse_float(match.group(2))

    if rate is None or amount is None:
        return None

    return {
        "rate": rate / 100,
        "amount": amount,
    }
# ============================================================
# Currency / locale
# ============================================================

def extract_currency(
    *texts: str | None,
) -> str | None:
    """
    Extract a valid ISO-style currency code.

    We only accept known currency codes, so values such as
    'VAT' are never interpreted as currencies.
    """
    currency_codes = {
        "USD",
        "EUR",
        "GBP",
        "CHF",
        "CAD",
        "AUD",
        "NZD",
        "JPY",
        "CNY",
        "INR",
        "AED",
        "SAR",
        "MAD",
        "SEK",
        "NOK",
        "DKK",
        "PLN",
        "CZK",
        "HUF",
        "RON",
        "BGN",
        "TRY",
        "BRL",
        "MXN",
        "ZAR",
    }
    for text in texts:
        if not text:
            continue
        matches = re.findall(
            r"\b([A-Z]{3})\b",
            text,
        )
        for code in reversed(matches):
            if code in currency_codes:
                return code
    return None


def infer_country(
    customer_address: dict[str, Any] | None,
    supplier_address: dict[str, Any] | None,
) -> str | None:
    """Infer locale country from parsed addresses."""
    if customer_address:
        country = customer_address.get("country")
        if country:
            return country
    if supplier_address:
        country = supplier_address.get("country")
        if country:
            return country
    return None


# ============================================================
# Document type
# ============================================================

def infer_document_type(
    annotation: dict[str, Any],
) -> str | None:
    """
    Infer document type from TITLE.
    Important:
        COMMERCIAL INVOICE
            -> invoice

    'invoice' is checked before 'commercial'.
    """
    title = get_first_text(
        annotation,
        "TITLE",
    )
    if not title:
        return None
    text = title.lower()
    if "invoice" in text:
        return "invoice"
    if "receipt" in text:
        return "receipt"
    if "credit note" in text:
        return "credit_note"
    if "debit note" in text:
        return "debit_note"
    if "commercial" in text:
        return "commercial"
    return None


# ============================================================
# Pattern-based extraction
# ============================================================
def extract_fatura_value(
    annotation: dict[str, Any],
    field_name: str,
    template_patterns: dict[str, Any] | None = None,
) -> str | None:
    """
    Extract a field using the induced template pattern when
    available, otherwise fall back to the original FATURA field.
    The direct field itself is always used as a fallback.
    """
    # --------------------------------------------------------
    # Try template pattern first
    # --------------------------------------------------------
    if template_patterns:
        pattern = template_patterns.get(
            field_name
        )
        if pattern:
            # Pattern stored directly as a string.
            if isinstance(pattern, str):
                try:
                    regex = re.compile(
                        pattern,
                        flags=re.IGNORECASE,
                    )
                    other_text = get_text(
                        annotation.get("OTHER")
                    )
                    if other_text:
                        match = regex.search(
                            other_text
                        )
                        if match:
                            if match.groups():
                                return clean_text(
                                    match.group(1)
                                )
                            return clean_text(
                                match.group(0)
                            )
                except re.error:
                    pass
            # Pattern stored in a dictionary.
            if isinstance(pattern, dict):
                regex_text = (
                    pattern.get("regex")
                    or pattern.get("pattern")
                    or pattern.get("value")
                )
                if isinstance(
                    regex_text,
                    str,
                ):
                    try:
                        regex = re.compile(
                            regex_text,
                            flags=re.IGNORECASE,
                        )
                        other_text = get_text(
                            annotation.get("OTHER")
                        )
                        if other_text:
                            match = regex.search(
                                other_text
                            )
                            if match:
                                if match.groups():
                                    return clean_text(
                                        match.group(1)
                                    )
                                return clean_text(
                                    match.group(0)
                                )
                    except re.error:
                        pass
    # --------------------------------------------------------
    # Fallback to original FATURA field
    # --------------------------------------------------------
    return get_text(
        annotation.get(field_name)
    )



# ============================================================
# Main conversion
# ============================================================
def convert_fatura_annotation(
    annotation: dict[str, Any],
    template_patterns: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Convert one FATURA annotation into the target extraction
    schema.
    """
    # --------------------------------------------------------
    # Complete target schema.
    #
    # Missing values remain explicitly null.
    # --------------------------------------------------------
    fields: dict[str, Any] = {
        "supplier_name": None,
        "supplier_phone_number": None,
        "supplier_address": empty_address(),
        "customer_name": None,
        "customer_phone_number": None,
        "customer_address": empty_address(),
        "invoice_number": None,
        "document_type": None,
        "date": None,
        "due_date": None,
        "period": None,
        "locale": {
            "language": None,
            "country": None,
            "currency": None,
        },
        "total_net": None,
        "total_tax": None,
        "total_amount": None,
        "taxes": [],
        "discount": None,
        "line_items": [],
    }
    # ========================================================
    # Supplier name
    # ========================================================
    supplier_name_text = get_first_text(
        annotation,
        "SELLER_NAME",
    )
    if supplier_name_text:
        fields["supplier_name"] = (
            supplier_name_text
        )
    # ========================================================
    # Supplier phone
    # ========================================================
    supplier_phone_text = get_first_text(
        annotation,
        "SELLER_PHONE",
        "SELLER_PHONE_NUMBER",
        "SUPPLIER_PHONE",
        "SUPPLIER_PHONE_NUMBER",
    )
    if supplier_phone_text:
        fields["supplier_phone_number"] = (
            supplier_phone_text
        )
    # ========================================================
    # Supplier address
    # ========================================================
    supplier_address_text = get_first_text(
        annotation,
        "SELLER_ADDRESS",
        "SUPPLIER_ADDRESS",
    )
    if supplier_address_text:
        parsed_supplier_address = (
            parse_seller_address_text(
                supplier_address_text
            )
        )
        if parsed_supplier_address:
            fields["supplier_address"] = (
                normalize_address_schema(
                    parsed_supplier_address
                )
            )
    # ========================================================
    # Customer / buyer
    # ========================================================
    # Priority:
    #
    # 1. BUYER
    # 2. BILL_TO
    # 3. SEND_TO / SHIP_TO only when no buyer is available
    #
    # SEND_TO / SHIP_TO is used as a customer fallback because
    # some FATURA templates provide shipping information instead
    # of a BUYER/BILL_TO field.
    # ========================================================
    buyer_text = get_first_text(
        annotation,
        "BUYER",
        "BILL_TO",
    )
    if buyer_text:
        (
            customer_name,
            customer_address,
            customer_phone_number,
        ) = parse_buyer_text(
            buyer_text
        )
        if customer_name:
            fields["customer_name"] = (
                customer_name
            )

        if customer_address:
            fields["customer_address"] = (
                normalize_address_schema(
                    customer_address
                )
            )
        if customer_phone_number:
            fields["customer_phone_number"] = (
                customer_phone_number
            )

    else:
        # No BUYER/BILL_TO available.
        # Fall back to SEND_TO / SHIP_TO.
        ship_to_text = get_first_text(
            annotation,
            "SEND_TO",
            "SHIP_TO",
        )

        if ship_to_text:
            (
                customer_name,
                customer_address,
                customer_phone_number,
            ) = parse_buyer_text(
                ship_to_text
            )

            if customer_name:
                fields["customer_name"] = (
                    customer_name
                )

            if customer_address:
                fields["customer_address"] = (
                    normalize_address_schema(
                        customer_address
                    )
                )

            if customer_phone_number:
                fields["customer_phone_number"] = (
                    customer_phone_number
                )
    # ========================================================
    # Invoice number
    # ========================================================
    # First try the standard NUMBER field.
    invoice_number_text = get_first_text(
        annotation,
        "NUMBER",
    )
    # FATURA Template1 uses PO_NUMBER instead of NUMBER.
    #
    # Therefore PO_NUMBER is used as the invoice-number
    # fallback for this dataset.
    if not invoice_number_text:
        invoice_number_text = get_first_text(
            annotation,
            "PO_NUMBER",
        )
    if invoice_number_text:
        fields["invoice_number"] = (
            parse_invoice_number_text(
                invoice_number_text
            )
        )
    # ========================================================
    # Document type
    # ========================================================
    fields["document_type"] = (
        infer_document_type(
            annotation
        )
    )
    # ========================================================
    # Date
    #
    # Use original FATURA field so formatting such as
    # 11-May-2013 is preserved.
    # ========================================================
    date_text = get_first_text(
        annotation,
        "DATE",
    )
    if date_text:
        fields["date"] = (
            parse_date_text(
                date_text
            )
        )
    # ========================================================
    # Due date
    #
    # Use original FATURA field so formatting such as
    # 21-May-2016 is preserved.
    # ========================================================
    due_date_text = get_first_text(
        annotation,
        "DUE_DATE",
    )
    if due_date_text:
        fields["due_date"] = (
            parse_due_date_text(
                due_date_text
            )
        )
    # ========================================================
    # Period
    # ========================================================
    period_text = get_first_text(
        annotation,
        "PERIOD",
    )
    if period_text:
        fields["period"] = period_text
    # ========================================================
    # Total net / subtotal
    # ========================================================
    subtotal_text = get_first_text(
        annotation,
        "SUB_TOTAL",
        "SUBTOTAL",
    )
    if subtotal_text:
        fields["total_net"] = (
            extract_numeric_field(
                subtotal_text
            )
        )
    # ========================================================
    # Total amount
    # ========================================================
    total_text = get_first_text(
        annotation,
        "TOTAL",
    )
    if total_text:
        fields["total_amount"] = (
            extract_numeric_field(
                total_text
            )
        )
    # ========================================================
    # Total tax
    #
    # Example:
    #
    # TAX:VAT (4.18%): 13.03 USD
    #
    # rate   = 4.18
    # amount = 13.03
    # ========================================================
    tax_text = get_first_text(
        annotation,
        "TAX",
        "TOTAL_TAX",
        "TOTAL TAX",
    )
    if tax_text:
        fields["total_tax"] = (
            extract_tax_amount(
                tax_text
            )
        )

    # ========================================================
    # Discount
    #
    # Example:
    #
    # DISCOUNT(1.85%): (-) 13.42
    #
    # rate   = 0.0185
    # amount = 13.42
    # ========================================================
    discount_text = get_first_text(
        annotation,
        "DISCOUNT",
    )

    if discount_text:
        discount = extract_discount(
            discount_text
        )
        if discount is not None:
            fields["discount"] = discount
    # ========================================================
    # Locale
    # ========================================================
    fields["locale"]["country"] = (
        infer_country(
            fields["customer_address"],
            fields["supplier_address"],
        )
    )
    # IMPORTANT:
    #
    # Do NOT pass tax_text here.
    #
    # Otherwise "VAT" could be incorrectly interpreted
    # as the currency.
    fields["locale"]["currency"] = extract_currency(
        total_text,
        subtotal_text,
        tax_text,
        get_first_text(annotation, "DISCOUNT"),
    )
    # Language cannot be reliably inferred from FATURA,
    # therefore it intentionally remains null.
    # ========================================================
    # Taxes
    # ========================================================
    if tax_text:
        tax_rate = extract_tax_rate(
            tax_text
        )
        if tax_rate is not None:
            fields["taxes"] = [
                {
                    "rate": tax_rate,
                    "base": fields["total_net"],
                    "amount": fields["total_tax"],
                }
            ]
    # ========================================================
    # GST taxes
    # ========================================================
    for field_name, value in annotation.items():
        if not field_name.upper().startswith("GST("):
            continue
        gst_text = get_text(value)
        if not gst_text:
            continue
        gst_tax = extract_gst_tax(
            gst_text
        )
        if gst_tax is None:
            continue
        fields["taxes"].append(
            {
                "rate": gst_tax["rate"],
                "base": fields["total_net"],
                "amount": gst_tax["amount"],
            }
        )
    # ========================================================
    # Line items
    # ========================================================
    # FATURA's TABLE field only provides bounding boxes.
    # It does not provide enough structured information to
    # reliably build the target line_items schema.
    #
    # Therefore line_items intentionally remains [].
    fields["line_items"] = []
    return {
        "fields": fields
    }


# ============================================================
# Dataset conversion
# ============================================================

def convert_dataset(
    annotations: dict[str, dict[str, Any]],
    template_patterns: dict[str, Any] | None = None,
    template_ids: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Convert an entire FATURA dataset.

    Parameters
    ----------
    annotations:
        Mapping:
            document_id -> FATURA annotation
    template_patterns:
        Mapping:
            template_id -> induced patterns
    template_ids:
        Mapping:
            document_id -> template_id
    """
    converted: dict[str, Any] = {}
    for doc_id, annotation in annotations.items():
        template_id = None
        if template_ids:
            template_id = template_ids.get(
                doc_id
            )
        patterns_for_doc = None
        if template_patterns and template_id:
            patterns_for_doc = (
                template_patterns.get(
                    template_id
                )
            )
        converted[doc_id] = (
            convert_fatura_annotation(
                annotation,
                template_patterns=patterns_for_doc,
            )
        )
    return converted


# ============================================================
# Save
# ============================================================

def save_json(
    data: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save converted dataset as formatted JSON."""
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ============================================================
# Standalone execution
# ============================================================

if __name__ == "__main__":
    print(
        "FATURA schema conversion module loaded successfully."
    )