// Mirrors the Pydantic `Receipt` from compare_outlines.py so the comparison is fair.
// Note: BAML does Schema-Aligned Parsing (model generates freely, BAML coerces the
// output to this shape) — there is NO token-level masking here, unlike outlines.
class Item {
  name string
  price float
}

class Receipt {
  store_name string
  date string?   // ground truth: 05/01/2026 — strict format checked in Python
  time string?   // ground truth: 13:06:09
  items Item[]
  total_ttc float
}

function ExtractReceipt(receipt_img: image) -> Receipt {
  client CustomOllama
  prompt #"
    {{ _.role("user") }}
    Extrais les informations de ce ticket de caisse.
    {{ receipt_img }}

    {{ ctx.output_format }}
  "#
}

// Run this in the BAML VSCode playground to smoke-test the function.
test ticket {
  functions [ExtractReceipt]
  args {
    receipt_img {
      file "../ticket_caisse.jpeg"
    }
  }
}
