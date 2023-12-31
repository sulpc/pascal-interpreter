'''
simple calculator
- no whitespace, one digit number

grammar:
expr                : INTEGER PLUS INTEGER
'''

class Token():
    def __init__(self, type, value):
        self.type  = type   # token type, like ID
        self.value = value  # token value, like 'x'

    def __str__(self):
        return f'Token({self.type}, {repr(self.value)})'

    def __repr__(self):
        return self.__str__()

# Token types
INTEGER         = 'INTEGER'
PLUS            = 'PLUS'        # +
EOF             = 'EOF'


class Interpreter():
    def __init__(self, text):
        self.text = text
        self.pos  = 0
        self.current_token = None

    ### Lexer

    def error(self):
        raise Exception('Error parsing input')

    def get_next_token(self):
        '''lexical analyzer, lexer, scanner, tokenizer

        breaking a sentence apart into tokens. One token one time.
        '''
        text = self.text

        # EOF
        if self.pos >= len(text):
            return Token(EOF, None)

        current_char = text[self.pos]

        if current_char.isdigit():
            token = Token(INTEGER, int(current_char))
            self.pos += 1
            return token

        if current_char == '+':
            token = Token(PLUS, current_char)
            self.pos += 1
            return token

        self.error()

    ### Parser & Interpreter

    def eat(self, token_type):
        '''verify the token type
        '''
        if self.current_token.type == token_type:
            self.current_token = self.get_next_token()
        else:
            self.error()

    def expr(self):
        '''calc expr

        expr : INTEGER PLUS INTEGER
        '''
        self.current_token = self.get_next_token()

        # left
        left = self.current_token
        self.eat(INTEGER)

        # op
        op = self.current_token
        self.eat(PLUS)

        # right
        right = self.current_token
        self.eat(INTEGER)

        # calc
        result = left.value + right.value
        return result


if __name__ == '__main__':
    while True:
        try:
            text = input('calc> ')
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue

        interpreter = Interpreter(text)
        result = interpreter.expr()
        print(result)
