import argparse

def parse_range(range_str):
    try:
        if '-' in range_str:
            start_str, end_str = range_str.split('-')

            if start_str and not start_str.isdigit():
                raise ValueError("Der Startwert enthält ungültige Zeichen. Es dürfen nur Zahlen verwendet werden.")

            if end_str and not end_str.isdigit():
                raise ValueError("Der Endwert enthält ungültige Zeichen. Es dürfen nur Zahlen verwendet werden.")

            start = int(start_str) if start_str else None
            end = int(end_str) if end_str else None

            if start is None and end is None:
                raise ValueError("Sowohl Start als auch Ende fehlen.")

            if start is None:
                start = 1  
            if end is None:
                end = 53

        else:
            if not range_str.isdigit():
                raise ValueError("Die Eingabe enthält ungültige Zeichen. Es dürfen nur Zahlen verwendet werden.")

            start = int(range_str)
            end = start 

        if start > end:
            raise ValueError("Der Startwert darf nicht größer als der Endwert sein.")

        return range(start, end + 1)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))

def parseArgs():
    parser = argparse.ArgumentParser(description="Anaglyph-Bild erstellen")
    parser.add_argument("-c", "--colorMode", type=str, default="color", help="Anaglyph-Color-Mode")
    parser.add_argument("-s", "--shift", type=bool, default=False, help="Shift-flag")
    parser.add_argument("-i", "--image", type=str, default="1", help="Image-Number")

    args = parser.parse_args()

    return parse_range(args.image), args.colorMode, args.shift

