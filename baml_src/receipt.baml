// Mirrors the Pydantic `Receipt` from compare_outlines.py (the SROIE ground-truth keys)
// so the comparison is fair — keep the field names, types and @description hints in sync.
// Note: BAML does Schema-Aligned Parsing (model generates freely, BAML coerces the
// output to this shape) — there is NO token-level masking here, unlike outlines.
class Receipt {
  company string @description("name of the store")
  date string @description("purchase date as printed")
  address string @description("full address of the store")
  total float @description("total amount paid, without currency symbol")
}

function ExtractReceipt(receipt_img: image) -> Receipt {
  client CustomOllama
  prompt #"
    {{ _.role("user") }}
    Extract the information from this receipt.
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
