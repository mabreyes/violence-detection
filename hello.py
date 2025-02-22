def main():
    class TuringMachine:
        """
        A Turing Machine implementation for palindrome detection.
        
        Diagram of the state transitions:
        
        [q0] ---> [q1] ---> [q2] ---> [q3]
         |         |         |         |
         |         |         |         |
         v         v         v         v
      [qaccept]   [q2]   [qaccept] [qreject]
                           |
                           v
                       [qreject]

        Tape transformation example for input "radar":
        Initial:    r  a  d  a  r  _
        Step 1:     R  a  d  a  r  _    (q0->q1: mark first 'r')
        Step 2:     R  a  d  a  R  _    (q1->q2: reach end, move left)
        Step 3:     R  a  d  A  *  _    (q2->q3: mark 'a', replace R with *)
        Step 4:     R  A  d  *  *  _    (q3->q2: match 'a', mark it)
        Step 5:     R  *  D  *  *  _    (q2->q3: mark 'd')
        Final:      *  *  *  *  *  _    (accepted)
        """
        def __init__(self):
            # Initialize empty tape, head position and states
            self.tape = []          # The tape holds input symbols
            self.head = 0           # Current position of read/write head
            self.state = 'q0'       # Initial state
            self.accepted_state = 'qaccept'  # Final accepting state
            self.rejected_state = 'qreject'  # Final rejecting state

        def write(self, symbol):
            """Write a symbol at current head position"""
            self.tape[self.head] = symbol

        def read(self):
            """Read symbol at current head position"""
            return self.tape[self.head]

        def move_left(self):
            """
            Move head left, but not beyond leftmost cell
            Tape boundary: [|0|1|2|3|...]
            """
            self.head = max(0, self.head - 1)

        def move_right(self):
            """
            Move head right, extending tape with blank if needed
            Tape expansion: [|a|b|c|_] -> [|a|b|c|_|_]
            """
            self.head += 1
            if self.head == len(self.tape):
                self.tape.append('_')  # Blank symbol

        def check_palindrome(self, input_str: str) -> bool:
            """
            Implements a Turing Machine that accepts palindromes.
            
            Algorithm steps:
            1. Start in q0, mark first character by capitalizing
            2. Move right to end of string (q1)
            3. Move left looking for marked character (q2)
            4. Compare current character with marked one (q3)
            5. Repeat until fully processed or mismatch found

            State transitions explained:
            q0 (Initial):
                - If tape empty -> Accept
                - Otherwise -> Mark first char, move right, goto q1
            
            q1 (Right scan):
                - If blank found -> Move left, goto q2
                - Otherwise -> Move right, stay in q1
            
            q2 (Left scan):
                - If at leftmost cell -> Accept
                - If marked char found -> Mark as processed (*), goto q3
                - Otherwise -> Move left, stay in q2
            
            q3 (Compare):
                - If blank found -> Reject
                - If marked char found -> Move left, goto q2
                - If chars match -> Mark current, move right, stay in q3
                - If mismatch -> Reject
            """
            # Initialize tape with input and blank symbol at end
            self.tape = list(input_str.lower()) + ['_']
            self.head = 0
            self.state = 'q0'

            while self.state not in [self.accepted_state, self.rejected_state]:
                if self.state == 'q0':
                    if self.read() == '_':  # Empty string is palindrome
                        self.state = self.accepted_state
                    else:
                        current = self.read()
                        self.write(current.upper())  # Mark by capitalizing
                        self.move_right()
                        self.state = 'q1'

                elif self.state == 'q1':
                    if self.read() == '_':
                        self.move_left()
                        self.state = 'q2'
                    else:
                        self.move_right()

                elif self.state == 'q2':
                    if self.head == 0:  # Single character left
                        self.state = self.accepted_state
                    elif self.read().isupper():  # Found marked character
                        current = self.read().lower()
                        self.write('*')  # Mark as processed
                        self.move_right()
                        self.state = 'q3'
                    else:
                        self.move_left()

                elif self.state == 'q3':
                    if self.read() == '_':  # Reached end too soon
                        self.state = self.rejected_state
                    elif self.read().isupper():  # Found another marked char
                        self.move_left()
                        self.state = 'q2'
                    else:
                        if self.read().lower() == current:  # Characters match
                            self.write(self.read().upper())  # Mark as processed
                            self.move_right()
                        else:  # Characters don't match
                            self.state = self.rejected_state

            return self.state == self.accepted_state

    # Create Turing Machine instance
    tm = TuringMachine()

    # Test cases demonstrating various palindrome patterns
    test_strings = [
        "radar",    # Classic palindrome
        "noon",     # Even length palindrome
        "hello",    # Non-palindrome
        "12321",    # Numeric palindrome
        "python",   # Non-palindrome
        "",         # Empty string (palindrome by definition)
        "a",        # Single character (palindrome)
    ]

    print("Turing Machine Palindrome Detection Results:")
    for text in test_strings:
        result = tm.check_palindrome(text)
        print(f"'{text}' -> {'✓' if result else '✗'}")


if __name__ == "__main__":
    main()
