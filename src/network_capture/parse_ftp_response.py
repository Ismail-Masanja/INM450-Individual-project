__all__ = ['parse_ftp_response']

def parse_ftp_response(payload: bytes) -> int:
    """
    Parses the FTP response from a given payload and extracts the response code.

    Parameters:
        payload (bytes): The payload received from an FTP server as a byte sequence.

    Returns:
        int: The first three-digit FTP response code found in the payload. If no valid code is found or an error occurs during parsing, returns 0.
    """
    try:
        # Decode payload from bytes to ASCII string
        text = payload.decode("ascii")
        # Split decoded text into lines
        lines = text.split('\r\n')
        
        for line in lines:
            if len(line) >= 3 and line[:3].isdigit():
                # Return first three-digit code found as integer
                return int(line[:3])
    except Exception as e:
        # In case of any error during decoding or parsing
        # Pass silenty. Log functionality is better.
        # print(f"Error parsing FTP response: {e}")
        pass
        
    # Default return value for no code
    return 0