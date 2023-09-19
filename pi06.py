'''
simple calculator, based on 5.py:
- add parenthesized expression

grammar:
expr                : term ((PLUS | MINUS) term)*
term                : factor ((MUL | DIV) factor)*
factor              : INTEGER
                    | LPAREN expr RPAREN
'''

###############################################################################
#                                                                             #
#  LEXER                                                                      #
#                                                                             #
###############################################################################

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
MINUS           = 'MINUS'       # -
MUL             = 'MUL'         # *
DIV             = 'DIV'         # /
LPAREN          = 'LPAREN'      # (
RPAREN          = 'RPAREN'      # )
EOF             = 'EOF'

class Lexer():
    def __init__(self, text: str):
        self.text = text
        self.pos  = 0
        # self.current_token = None
        self.current_char  = self.text[self.pos]

    def error(self):
        raise Exception('Error parsing input')

    def advance(self):
        '''get next char, and increse the the pos pointer

        advance the 'pos' pointer and set the 'current_char' variable.
        '''
        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None  # end of input
        else:
            self.current_char = self.text[self.pos]

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def integer(self):
        '''parse an integer from the input
        '''
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()

        return int(result)

    def get_next_token(self):
        '''lexical analyzer, lexer, scanner, tokenizer

        breaking a sentence apart into tokens. One token one time.
        '''
        while self.current_char is not None:
            # space
            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            # digit -> integer
            if self.current_char.isdigit():
                return Token(INTEGER, self.integer())

            if self.current_char == '+':
                self.advance()
                return Token(PLUS, '+')

            if self.current_char == '-':
                self.advance()
                return Token(MINUS, '-')

            if self.current_char == '*':
                self.advance()
                return Token(MUL, '*')

            if self.current_char == '/':
                self.advance()
                return Token(DIV, '/')

            if self.current_char == '(':
                self.advance()
                return Token(LPAREN, '(')

            if self.current_char == ')':
                self.advance()
                return Token(RPAREN, ')')

            self.error()

        return Token(EOF, None)

###############################################################################
#                                                                             #
#  INTERPRETER                                                                #
#                                                                             #
###############################################################################
class Interpreter():
    def __init__(self, lexer: Lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()

    def error(self):
        raise Exception('Invalid syntax')

    def eat(self, token_type):
        '''verify the token type
        '''
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error()

    def expr(self):
        '''calc expr

        expr : term ((PLUS | MINUS) term)*
        '''
        result = self.term()

        while self.current_token.type in (PLUS, MINUS):
            op = self.current_token
            if op.type == PLUS:
                self.eat(PLUS)
                result = result + self.term()
            else:
                self.eat(MINUS)
                result = result - self.term()

        return result

    def term(self):
        '''calc term

        term : factor ((MUL | DIV) factor)*
        '''
        result = self.factor()

        while self.current_token.type in (MUL, DIV):
            op = self.current_token
            if op.type == MUL:
                self.eat(MUL)
                result = result * self.factor()
            else:
                self.eat(DIV)
                result = result / self.factor()

        return result

    def factor(self):
        '''calc factor

        factor : INTEGER
               | LPAREN expr RPAREN
        '''
        token = self.current_token
        if token.type == INTEGER:
            self.eat(INTEGER)
            return token.value
        elif token.type == LPAREN:
            self.eat(LPAREN)
            result = self.expr()
            self.eat(RPAREN)
            return result


if __name__ == '__main__':
    while True:
        try:
            text = input('calc> ')
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue

        lexer = Lexer(text)
        interpreter = Interpreter(lexer)
        result = interpreter.expr()
        print(result)
