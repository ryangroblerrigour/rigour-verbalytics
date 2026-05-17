import re
import httpx


POSTCODE_REGEX = re.compile(
    r"^[A-Z]{1,2}[0-9][0-9A-Z]?\s?[0-9][A-Z]{2}$"
)

OUTCODE_REGEX = re.compile(
    r"^[A-Z]{1,2}[0-9][0-9A-Z]?$"
)

IMPLAUSIBLE_POSTCODES = {
    "SW1A 1AA": "Buckingham Palace",
    "SW1A 2AA": "10 Downing Street",
}
async def lookup_postcode(postcode: str):
    postcode = postcode.strip().upper()

    is_full = bool(POSTCODE_REGEX.match(postcode))
    is_partial = bool(OUTCODE_REGEX.match(postcode))

    if not is_full and not is_partial:
        return {
            "is_valid": False,
            "message": "Invalid postcode format"
        }

    if is_full:
        url = f"https://api.postcodes.io/postcodes/{postcode}"

    else:
        url = f"https://api.postcodes.io/outcodes/{postcode}"

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(url)

    data = response.json()

    if response.status_code != 200:
        return {
            "is_valid": False,
            "message": "Postcode not found"
        }

    result = data.get("result", {})
    
        implausible_reason = IMPLAUSIBLE_POSTCODES.get(
        result.get("postcode") or postcode
    )

    return {
        "is_valid": True,
        "is_partial": is_partial,
        "postcode": postcode,
        "country": result.get("country"),
        "region": result.get("region") or result.get("country"),
        "admin_district": result.get("admin_district"),
        "urban_rural": result.get("rural_urban") or result.get("codes", {}).get("rural_urban"),
        "implausible": bool(implausible_reason),
        "implausible_reason": implausible_reason,
    }
