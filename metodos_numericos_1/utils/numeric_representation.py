# utils/numeric_representation.py

## Mantissa tem que ser maior que 1/base e menor que 1

def dec_to_oct(n: int, signed=True) -> str:
    """Convert a decimal number to octal representation."""
    if n < 0 and not signed:
        raise ValueError("Negative numbers are not supported in unsigned mode.")
    return oct(n).replace("0o", "").upper() if signed else oct(n & 0o77777777).replace("0o", "").upper()


def dec_to_hex(n: int, signed=True) -> str:
    """Convert a decimal number to hexadecimal representation."""
    if n < 0 and not signed:
        raise ValueError("Negative numbers are not supported in unsigned mode.")
    return hex(n).replace("0x", "").upper() if signed else hex(n & 0xFFFFFFFF).replace("0x", "").upper()


def dec_to_bin(n, signed=True, floating_point=False) -> str:
    """
    Convert a decimal number to binary representation.

    Parameters
    ----------
    n : int or float
        The decimal number to convert
    signed : bool, default=True
        Whether to treat integers as signed
    floating_point : bool, default=False
        Whether to treat the input as a floating point number

    Returns
    -------
    str
        Binary representation of the number
    """
    if floating_point:
        if n == 0:
            return "0.0"

        # Handle negative numbers
        sign_bit = "1" if n < 0 else "0"
        n = abs(n)

        # Split into integer and fractional parts
        int_part = int(n)
        frac_part = n - int_part

        # Convert integer part to binary
        int_binary = bin(int_part).replace("0b", "")

        # Convert fractional part to binary (limited precision)
        frac_binary = ""
        precision = 23  # Single precision mantissa

        while frac_part > 0 and len(frac_binary) < precision:
            frac_part *= 2
            bit = int(frac_part)
            frac_binary += str(bit)
            frac_part -= bit

        # For simplicity, we're returning a simple representation
        if frac_binary:
            return f"{sign_bit}_{int_binary}.{frac_binary}"
        else:
            return f"{sign_bit}_{int_binary}.0"

    else:
        # Integer handling
        if n < 0 and not signed:
            raise ValueError("Negative numbers are not supported in unsigned mode.")

        if signed:
            return bin(n).replace("0b", "")
        else:
            return bin(n & 0xFFFFFFFF).replace("0b", "")


def bin_to_dec(binary_str: str, signed=True) -> int:
    """
    Convert a binary string to decimal representation.

    Parameters
    ----------
    binary_str : str
        The binary string to convert
    signed : bool, default=True
        Whether to treat the binary string as signed

    Returns
    -------
    int
        Decimal representation of the binary string
    """
    if signed:
        return int(binary_str, 2)
    else:
        return int(binary_str, 2) & 0xFFFFFFFF  # Mask for unsigned conversion


if __name__ == "__main__":
    # Example usage
    print("Decimal to Octal (signed):", dec_to_oct(105))
    print("Decimal to Hexadecimal (unsigned):", dec_to_hex(105))
    print("Decimal to Binary (signed):", dec_to_bin(105))
    print("Decimal to Binary (floating point):", dec_to_bin(-3.14, floating_point=True))
    print("Binary to Decimal (signed):", bin_to_dec("01101001"))
    print("Binary to Decimal (unsigned):", bin_to_dec("11111111", signed=False))