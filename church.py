class ChurchTuringMachine:
    """
    A Turing Machine implementation for Church numeral addition of any two numbers.

    Church numerals represent numbers through function application:
    0 := λf.λx.x
    1 := λf.λx.f(x)
    2 := λf.λx.f(f(x))
    n := λf.λx.f^n(x)  # f applied n times

    State Transition Diagram:

    [q0] -----|---> [q1] -----|---> [q2] ---_---> [q3]
                      |          |              |
                      |          |              |
                      v          v              v
                    [q2]      [q3]         [qaccept]
                    (on #)    (on #)

    Tape Transformation Example (2+3):
    Initial:    _ | | # | | | _
    Step 1:     _ * * # | | | _    (Mark first number)
    Step 2:     _ * * # * * * _    (Mark second number)
    Step 3:     _ | | | | | _ _    (Combine marks)
    Final:      _ | | | | | _ _    (Result shows 5 in Church form)

    Symbols:
    | - Represents one unit
    # - Separator between numbers
    * - Temporary marking
    _ - Blank/empty cell
    """

    def __init__(self):
        self.tape = []
        self.head = 0
        self.state = "q0"
        self.accepted_state = "qaccept"

    def write(self, symbol):
        """Write a symbol at current head position"""
        self.tape[self.head] = symbol

    def read(self):
        """Read symbol at current head position"""
        return self.tape[self.head]

    def move_right(self):
        """Move head right, extending tape if needed"""
        self.head += 1
        if self.head == len(self.tape):
            self.tape.append("_")

    def move_left(self):
        """Move head left, not beyond leftmost cell"""
        self.head = max(0, self.head - 1)

    def add_numbers(self, n1: int, n2: int):
        """
        Implements Church numeral addition for any two numbers.

        Algorithm Steps:
        1. Initialize tape with unary representation: n1 # n2
        2. q0: Mark first number (convert | to *)
        3. q1: Find and mark second number
        4. q2: Convert all marks to final representation
        5. q3: Clean up and prepare output

        Args:
            n1 (int): First number to add
            n2 (int): Second number to add

        Returns:
            str: Final tape contents showing result in Church numeral form
        """
        # Initialize tape with input numbers separated by #
        self.tape = ["_"] + ["|"] * n1 + ["#"] + ["|"] * n2 + ["_"]
        self.head = 0
        self.state = "q0"

        while self.state != self.accepted_state:
            if self.state == "q0":
                if self.read() == "|":
                    # Mark first number
                    self.write("*")
                    self.move_right()
                elif self.read() == "#":
                    # Found separator, move to second number
                    self.move_right()
                    self.state = "q1"
                else:
                    self.move_right()

            elif self.state == "q1":
                if self.read() == "|":
                    # Mark second number
                    self.write("*")
                    self.move_right()
                elif self.read() == "_":
                    # Reached end, start combining
                    self.move_left()
                    self.state = "q2"
                else:
                    self.move_right()

            elif self.state == "q2":
                if self.read() == "*":
                    # Convert marks back to |
                    self.write("|")
                    self.move_left()
                elif self.read() == "#":
                    # Skip separator
                    self.move_left()
                elif self.read() == "_":
                    # Finished combining
                    self.state = "q3"
                else:
                    self.move_left()

            elif self.state == "q3":
                if self.read() == "_":
                    self.state = self.accepted_state
                else:
                    self.move_right()

        # Return final tape contents (stripped of leading/trailing blanks)
        return "".join(symbol for symbol in self.tape if symbol == "|")


def main():
    """
    Main function demonstrating Church numeral addition.
    Tests various number combinations and displays results.
    """
    # Create Church TM instance
    tm = ChurchTuringMachine()

    # Test cases
    test_pairs = [
        (1, 1),  # Basic case
        (2, 3),  # Larger numbers
        (0, 4),  # Zero case
        (3, 0),  # Zero case reversed
        (5, 5),  # Equal numbers
    ]

    print("Church Numeral Addition Examples:")
    for n1, n2 in test_pairs:
        result = tm.add_numbers(n1, n2)
        print(f"\n{n1} + {n2}:")
        print(f"Church representation: {result}")
        print(f"Decimal value: {result.count('|')}")
        print(f"Lambda calculus: λf.λx.f^{result.count('|')}(x)")


if __name__ == "__main__":
    main()
