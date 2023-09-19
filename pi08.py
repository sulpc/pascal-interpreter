'''
simple calculator, based on 7.py:
- unary operators

grammar:
expr                : term ((PLUS | MINUS) term)*
term                : factor ((MUL | DIV) factor)*
factor              : (PLUS | MINUS) factor
                    | INTEGER
                    | LPAREN expr RPAREN
'''

DISPLAY_AST   = True
LOCAL_ECHARTS = True

if DISPLAY_AST:
    from pyecharts import options as opts
    from pyecharts.charts import Tree
    import os

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
#  PARSER                                                                     #
#                                                                             #
###############################################################################

class AST():
    pass

### expr

class Num(AST):
    '''
    '''
    def __init__(self, token: Token):
        self.token = token
        self.value = token.value

class UnaryOp(AST):
    '''
    '''
    def __init__(self, op: Token, expr):
        self.op    = op
        self.token = self.op
        self.expr  = expr

class BinOp(AST):
    '''
    '''
    def __init__(self, left, op: Token, right):
        self.left  = left
        self.op    = op
        self.right = right
        self.token = self.op

class Parser():
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
        '''parse expr

        expr: term ((PLUS | MINUS) term)*
        '''
        result = self.term()

        while self.current_token.type in (PLUS, MINUS):
            op = self.current_token
            if op.type == PLUS:
                self.eat(PLUS)
            else:
                self.eat(MINUS)
            result = BinOp(left=result, op=op, right=self.term())

        return result

    def term(self):
        '''parse term

        term: factor ((MUL | DIV) factor)*
        '''
        result = self.factor()

        while self.current_token.type in (MUL, DIV):
            op = self.current_token
            if op.type == MUL:
                self.eat(MUL)
            elif op.type == DIV:
                self.eat(DIV)
            result = BinOp(left=result, op=op, right=self.factor())

        return result

    def factor(self):
        '''parse factor

        factor : (PLUS | MINUS) factor
               | INTEGER
               | LPAREN expr RPAREN
        '''
        token = self.current_token
        if token.type == PLUS:
            self.eat(PLUS)
            result = UnaryOp(token, self.factor())
            return result
        elif token.type == MINUS:
            self.eat(MINUS)
            result = UnaryOp(token, self.factor())
            return result
        elif token.type == INTEGER:
            self.eat(INTEGER)
            return Num(token)
        elif token.type == LPAREN:
            self.eat(LPAREN)
            result = self.expr()
            self.eat(RPAREN)
            return result

    def parse(self):
        return self.expr()

###############################################################################
#                                                                             #
#  INTERPRETER                                                                #
#                                                                             #
###############################################################################

class NodeVistor():
    def visit(self, node):
        '''dispatches
        '''
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visitor)
        return visitor(node)

    def generic_visitor(self, node):
        raise Exception(f'No visit_{type(node).__name__} method')

    def pack(self, node):
        '''dispatches: pack data to display
        '''
        method_name = 'pack_' + type(node).__name__
        packer = getattr(self, method_name, self.generic_packer)
        return packer(node)

    def generic_packer(self, node):
        raise Exception(f'No pack_{type(node).__name__} method')


class Interpreter(NodeVistor):
    def __init__(self, parser: Parser) -> None:
        self.parser = parser

    def visit_BinOp(self, node: BinOp):
        if node.op.type == PLUS:
            return self.visit(node.left) + self.visit(node.right)
        elif node.op.type == MINUS:
            return self.visit(node.left) - self.visit(node.right)
        elif node.op.type == MUL:
            return self.visit(node.left) * self.visit(node.right)
        elif node.op.type == DIV:
            return self.visit(node.left) / self.visit(node.right)

    def visit_UnaryOp(self, node: UnaryOp):
        if node.op.type == PLUS:
            return self.visit(node.expr)
        else:   # node.op.type == MINUS
            return -self.visit(node.expr)

    def visit_Num(self, node: Num):
        return node.value

    # packup: pack_*

    def pack_BinOp(self, node: BinOp):
        data = {
            'name': f'{node.op.value}',
            'children': [self.pack(node.left), self.pack(node.right)]
        }
        return data

    def pack_UnaryOp(self, node: UnaryOp):
        data = {
            'name': f'{node.op.value}',
            'children': [self.pack(node.expr)]
        }
        return data

    def pack_Num(self, node: Num):
        data = {
            'name': f'{str(node.value)}'
        }
        return data

    def interpret(self):
        tree = self.parser.parse()

        if DISPLAY_AST:
            data = self.pack(tree)
            c = (
                Tree()
                .add(
                    series_name="",         # name
                    data=[data],            # data
                    initial_tree_depth=-1,  # all expand
                    orient="TB",            # top-to-bottom
                    label_opts=opts.LabelOpts(
                        position="top",
                        vertical_align="middle",
                    ),
                )
                .set_global_opts(title_opts=opts.TitleOpts(title="Tree"))
                .render('html')
            )
            # modify js reference to local
            with open('html', 'r') as fin:
                content = fin.readlines()
                content[4] = '    <title>Tree</title>\n'
                if LOCAL_ECHARTS:
                    content[5] = '    <script type="text/javascript" src="echarts.min.js"></script>\n'
            with open('Tree.html', 'w') as fout:
                fout.writelines(content)
            os.remove('html')

        # interprete
        return self.visit(tree)


if __name__ == '__main__':
    while True:
        try:
            text = input('calc> ')
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue

        lexer = Lexer(text)
        parser = Parser(lexer)
        interpreter = Interpreter(parser)
        result = interpreter.interpret()
        print(result)
