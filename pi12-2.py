'''
simple pascal interpreter, based on 12-1.py
- modify code structure

grammar:
program             : PROGRAM variable SEMI block DOT
block               : declarations compound_statement
declarations        : VAR (variable_declaration SEMI)+
                    | (PROCEDURE ID SEMI block SEMI)*
                    | empty
variable_declaration: ID (COMMA ID)* COLON type_spec
type_spec           : INTEGER | REAL
compound_statement  : BEGIN statement_list END
statement_list      : statement
                    | statement SEMI statement_list
statement           : compound_statement
                    | assignment_statement
                    | empty
assignment_statement: variable ASSIGN expr
empty               :
expr                : term ((PLUS | MINUS) term)*
term                : factor ((MUL | INTEGER_DIV | FLOAT_DIV) factor)*
factor              : PLUS factor
                    | MINUS factor
                    | INTEGER_CONST
                    | REAL_CONST
                    | LPAREN expr RPAREN
                    | variable
variable            : ID
'''

test_code = '''
PROGRAM Part12;
VAR
   a : INTEGER;

PROCEDURE P1;
VAR
   a : REAL;
   k : INTEGER;

   PROCEDURE P2;
   VAR
      a, z : INTEGER;
   BEGIN {P2}
      z := 777;
   END;  {P2}

BEGIN {P1}

END;  {P1}

BEGIN {Part12}
   a := 10;
END.  {Part12}
'''

LOCAL_ECHARTS = True

from pyecharts import options as opts
from pyecharts.charts import Tree
import os
from collections import OrderedDict

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
ID              = 'ID'
INTEGER_CONST   = 'INTEGER_CONST'
REAL_CONST      = 'REAL_CONST'
PLUS            = 'PLUS'        # +
MINUS           = 'MINUS'       # -
MUL             = 'MUL'         # *
INTEGER_DIV     = 'INTEGER_DIV' # DIV
FLOAT_DIV       = 'FLOAT_DIV'   # /
LPAREN          = 'LPAREN'      # (
RPAREN          = 'RPAREN'      # )
DOT             = 'DOT'         # .
ASSIGN          = 'ASSIGN'      # :=
SEMI            = 'SEMI'        # ;
COMMA           = 'COMMA'       # ,
COLON           = 'COLON'       # :
EOF             = 'EOF'

# keywords
PROGRAM         = 'PROGRAM'
VAR             = 'VAR'
PROCEDURE       = 'PROCEDURE'
DIV             = 'DIV'
INTEGER         = 'INTEGER'
REAL            = 'REAL'
BEGIN           = 'BEGIN'
END             = 'END'
RESERVED_KEYWORDS = {
    'PROGRAM'   : Token(PROGRAM,    'PROGRAM'),
    'VAR'       : Token(VAR,        'VAR'),
    'PROCEDURE' : Token(PROCEDURE,  'PROCEDURE'),
    'DIV'       : Token(INTEGER_DIV,'DIV'),
    'INTEGER'   : Token(INTEGER,    'INTEGER'),
    'REAL'      : Token(REAL,       'REAL'),
    'BEGIN'     : Token(BEGIN,      'BEGIN'),
    'END'       : Token(END,        'END'),
}

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

    def peek(self):
        '''lookup next char, but not increse the pos pointer
        '''
        peek_pos = self.pos + 1
        if peek_pos > len(self.text) - 1:
            return None  # end of input
        else:
            return self.text[peek_pos]

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def skip_comment(self):
        while self.current_char != '}':
            self.advance()
        self.advance()  # eat '}'

    def number(self):
        '''parse an number from the input
        '''
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()

        if self.current_char != '.':
            token = Token(INTEGER_CONST, int(result))
        else:
            result += self.current_char
            self.advance()

            while self.current_char is not None and self.current_char.isdigit():
                result += self.current_char
                self.advance()

            token = Token(REAL_CONST, float(result))

        return token

    def _id(self):
        '''parse an identifier or reserved keyword
        '''
        result = ''
        while self.current_char is not None and self.current_char.isalnum():
            result += self.current_char
            self.advance()

        token = RESERVED_KEYWORDS.get(result, Token(ID, result))
        return token

    def get_next_token(self):
        '''lexical analyzer, lexer, scanner, tokenizer

        breaking a sentence apart into tokens. One token one time.
        '''
        while self.current_char is not None:
            # space
            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char == '{':
                self.skip_comment()
                continue

            # digit -> number
            if self.current_char.isdigit():
                return self.number()

            # alpha -> id/keyword
            if self.current_char.isalpha():
                return self._id()

            # :=
            if self.current_char == ':':
                if self.peek() == '=':
                    self.advance()
                    self.advance()
                    return Token(ASSIGN, ':=')
                else:
                    self.advance()
                    return Token(COLON, ':')

            if self.current_char == ',':
                self.advance()
                return Token(COMMA, ',')

            if self.current_char == ';':
                self.advance()
                return Token(SEMI, ';')

            if self.current_char == '.':
                self.advance()
                return Token(DOT, '.')

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
                return Token(FLOAT_DIV, '/')

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

class Var(AST):
    '''variable
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

### stat

class NoOp(AST):
    '''empty
    '''
    pass

class Assign(AST):
    '''assignment_statement
    '''
    def __init__(self, left: Token, op: Token, right: Token):
        self.left = left
        self.token = self.op = op
        self.right = right

class Compound(AST):
    '''compound_statement & statement_list

    children: statement_list
    '''
    def __init__(self):
        self.children = []

class Type(AST):
    '''type_spec
    '''
    def __init__(self, token: Token):
        self.token = token
        self.value = token.value

class VarDecl(AST):
    '''one var decl

    for variable_declaration
    '''
    def __init__(self, var_node: Token, type_node: Type):
        self.var_node = var_node
        self.type_node = type_node

    def __str__(self):
        return f'{self.var_node.value, self.type_node.value}'

class Block(AST):
    '''block
    '''
    def __init__(self, declarations: list, compound_statement: Compound):
        self.declarations = declarations
        self.compound_statement = compound_statement

class ProcedureDecl(AST):
    '''procedure
    '''
    def __init__(self, proc_name, block_node: Block) -> None:
        self.proc_name = proc_name
        self.block_node = block_node

class Program(AST):
    '''program
    '''
    def __init__(self, name, block: Block):
        self.name = name
        self.block = block


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

    def program(self):
        '''parse program

        program : PROGRAM variable SEMI block DOT
        '''
        self.eat(PROGRAM)
        prog_name = self.variable().value
        self.eat(SEMI)
        block_node = self.block()
        self.eat(DOT)
        return Program(prog_name, block_node)

    def block(self):
        '''parse block

        block : declarations compound_statement
        '''
        declarations_node = self.declarations()
        compound_statement_node = self.compound_statement()
        return Block(declarations_node, compound_statement_node)

    def declarations(self):
        '''parse declarations

        declarations : VAR (variable_declaration SEMI)+
                     | (PROCEDURE ID SEMI block SEMI)*
                     | empty
        '''
        declarations_node = []
        if self.current_token.type == VAR:
            self.eat(VAR)
            while self.current_token.type == ID:
                var_decl = self.variable_declaration()
                declarations_node.extend(var_decl)  # extend 添加可迭代元素
                self.eat(SEMI)

        while self.current_token.type == PROCEDURE:
            self.eat(PROCEDURE)
            proc_name = self.current_token.value
            self.eat(ID)
            self.eat(SEMI)
            block_node = self.block()
            self.eat(SEMI)
            proc_decl = ProcedureDecl(proc_name, block_node)
            declarations_node.append(proc_decl)

        return declarations_node

    def variable_declaration(self):
        '''parse variable_declaration

        variable_declaration: ID (COMMA ID)* COLON type_spec
        '''
        var_nodes = [Var(self.current_token)]
        self.eat(ID)

        while self.current_token.type == COMMA:
            self.eat(COMMA)
            var_nodes.append(Var(self.current_token))
            self.eat(ID)

        self.eat(COLON)
        type_node = self.type_spec()
        return [VarDecl(var_node, type_node) for var_node in var_nodes]

    def type_spec(self):
        '''parse type_spec

        type_spec : INTEGER | REAL
        '''
        token = self.current_token
        if token.type == INTEGER:
            self.eat(INTEGER)
        else:
            self.eat(REAL)
        return Type(token)

    def compound_statement(self):
        '''parse compound_statement

        compound_statement : BEGIN statement_list END
        '''
        self.eat(BEGIN)
        stats = self.statement_list()
        self.eat(END)

        result = Compound()
        for stat in stats:
            result.children.append(stat)

        return result

    def statement_list(self):
        '''parse statement_list

        statement_list : statement
                       | statement SEMI statement_list
        '''
        stat   = self.statement()
        result = [stat]

        while self.current_token.type == SEMI:
            self.eat(SEMI)
            result.append(self.statement())

        # TODO
        if self.current_token.type == ID:
            self.error()

        return result

    def statement(self):
        '''parse statement

        statement : compound_statement
                  | assignment_statement
                  | empty
        '''
        if self.current_token.type == BEGIN:
            result = self.compound_statement()
        elif self.current_token.type == ID:
            result = self.assignment_statement()
        else:
            result = self.empty()
        return result

    def assignment_statement(self):
        '''parse assignment_statement

        assignment_statement : variable ASSIGN expr
        '''
        left = self.variable()
        token = self.current_token
        self.eat(ASSIGN)
        right = self.expr()

        result = Assign(left, token, right)
        return result

    def empty(self):
        '''parse empty
        '''
        return NoOp()

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

        term: factor ((MUL | INTEGER_DIV | FLOAT_DIV) factor)*
        '''
        result = self.factor()

        while self.current_token.type in (MUL, INTEGER_DIV, FLOAT_DIV):
            op = self.current_token
            if op.type == MUL:
                self.eat(MUL)
            elif op.type == INTEGER_DIV:
                self.eat(INTEGER_DIV)
            elif op.type == FLOAT_DIV:
                self.eat(FLOAT_DIV)
            result = BinOp(left=result, op=op, right=self.factor())

        return result

    def factor(self):
        '''parse factor

        factor : PLUS factor
               | MINUS factor
               | INTEGER_CONST
               | REAL_CONST
               | LPAREN expr RPAREN
               | variable
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
        elif token.type == INTEGER_CONST:
            self.eat(INTEGER_CONST)
            return Num(token)
        elif token.type == REAL_CONST:
            self.eat(REAL_CONST)
            return Num(token)
        elif token.type == LPAREN:
            self.eat(LPAREN)
            result = self.expr()
            self.eat(RPAREN)
            return result
        else:
            return self.variable()

    def variable(self):
        '''parse variable
        '''
        result = Var(self.current_token)
        self.eat(ID)
        return result

    def parse(self):
        result = self.program()
        if self.current_token.type != EOF:
            self.error()
        return result

###############################################################################
#                                                                             #
#  SEMANTIC DISPLAY INTERPRETER                                               #
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

###############################################################################
#                                                                             #
#  DISPLAYER                                                                  #
#                                                                             #
###############################################################################

class Displayer(NodeVistor):
    def __init__(self, tree) -> None:
        self.tree          = tree

    def visit_Program(self, node: Program):
        data = {
            'name': f'{node.name}',
            'children': [self.visit(node.block)]
        }
        return data

    def visit_Block(self, node: Block):
        decls = {
            'name': 'Decls',
            'children': []
        }
        for decl in node.declarations:
            decls['children'].append(self.visit(decl))

        data = {
            'name': 'Block',
            'children': [decls, self.visit(node.compound_statement)]
        }
        return data

    def visit_ProcedureDecl(self, node: ProcedureDecl):
        data = {
            'name': f'Proc-{node.proc_name}',
            'children': [self.visit(node.block_node)]
        }
        return data

    def visit_VarDecl(self, node: VarDecl):
        data = {
            'name': 'VarDecl',
            'children': [self.visit(node.var_node), self.visit(node.type_node)]
        }
        return data

    def visit_Type(self, node: Type):
        data = {
            'name': f'{node.value}',
        }
        return data

    def visit_Compound(self, node: Compound):
        data = {
            'name': 'Compound',
            'children': []
        }
        for stat in node.children:
            data['children'].append(self.visit(stat))
        return data

    def visit_Assign(self, node: Assign):
        data = {
            'name': ':=',
            'children': []
        }
        data['children'].append(self.visit(node.left))
        data['children'].append(self.visit(node.right))
        return data

    def visit_NoOp(self, node: NoOp):
        data = {
            'name': 'NoOp',
        }
        return data

    def visit_BinOp(self, node: BinOp):
        data = {
            'name': f'{node.op.value}',
            'children': [self.visit(node.left), self.visit(node.right)]
        }
        return data

    def visit_UnaryOp(self, node: UnaryOp):
        data = {
            'name': f'{node.op.value}',
            'children': [self.visit(node.expr)]
        }
        return data

    def visit_Var(self, node: Var):
        data = {
            'name': f'{node.value}',
        }
        return data

    def visit_Num(self, node: Num):
        data = {
            'name': f'{str(node.value)}'
        }
        return data

    def display(self):
        tree = self.tree

        data = self.visit(tree)
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

###############################################################################
#                                                                             #
#  SEMANTIC ANALYZER                                                          #
#                                                                             #
###############################################################################

class Symbol():
    def __init__(self, name, type=None):
        self.name = name
        self.type = type

class BuiltinTypeSymbol(Symbol):
    def __init__(self, name):
        super().__init__(name)

    def __str__(self) -> str:
        return self.name

    __repr__= __str__

class VarSymbol(Symbol):
    def __init__(self, name, type):
        super().__init__(name, type)

    def __str__(self) -> str:
        return f'<{self.name}:{self.type}>'

    __repr__ = __str__

class SymbolTable():
    def __init__(self) -> None:
        self._symbols = OrderedDict()
        self._init_builtins()

    def __str__(self) -> str:
        return f'Symbols: {[val for val in self._symbols.values()]}'

    __repr__= __str__

    def _init_builtins(self):
        self.define(BuiltinTypeSymbol('INTEGER'))
        self.define(BuiltinTypeSymbol('REAL'))

    def define(self, symbol: Symbol):
        print(f'define: {symbol}')
        self._symbols[symbol.name] = symbol

    def lookup(self, name):
        print(f'lookup: {name}')
        # return a Symbol found or None
        return self._symbols.get(name)

class SymbolTableBuilder(NodeVistor):
    def __init__(self) -> None:
        self.symtab = SymbolTable()

    def visit_Program(self, node: Program):
        self.visit(node.block)

    def visit_Block(self, node: Block):
        for declaration in node.declarations:
            self.visit(declaration)
        self.visit(node.compound_statement)

    def visit_ProcedureDecl(self, node: ProcedureDecl):
        pass

    def visit_VarDecl(self, node: VarDecl):
        type_syb = self.symtab.lookup(node.type_node.value)
        var_syb = VarSymbol(node.var_node.value, type_syb)
        self.symtab.define(var_syb)

    def visit_Type(self, node: Type):
        pass

    def visit_Compound(self, node: Compound):
        for stat in node.children:
            self.visit(stat)

    def visit_Assign(self, node: Assign):
        '''check if left var has defined
        '''
        var_name = node.left.value
        var_sybl = self.symtab.lookup(var_name)
        if var_sybl is None:
            raise NameError(repr(var_name))
        else:
            self.visit(node.right)

    def visit_NoOp(self, node: NoOp):
        pass

    def visit_BinOp(self, node: BinOp):
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: UnaryOp):
        self.visit(node.expr)

    def visit_Var(self, node: Var):
        '''check var has defined
        '''
        var_name = node.value
        var_sybl = self.symtab.lookup(var_name)
        if var_sybl is None:
            raise NameError(repr(var_name))

    def visit_Num(self, node: Num):
        pass

###############################################################################
#                                                                             #
#   INTERPRETER                                                               #
#                                                                             #
###############################################################################

class Interpreter(NodeVistor):
    def __init__(self, tree) -> None:
        self.tree          = tree
        self.GLOBAL_MEMORY = {}  # symbol table

    def visit_Program(self, node: Program):
        self.visit(node.block)

    def visit_Block(self, node: Block):
        for declaration in node.declarations:
            self.visit(declaration)
        self.visit(node.compound_statement)

    def visit_ProcedureDecl(self, node: ProcedureDecl):
        pass

    def visit_VarDecl(self, node: VarDecl):
        # TODO
        pass

    def visit_Type(self, node: Type):
        # TODO
        pass

    def visit_Compound(self, node: Compound):
        for stat in node.children:
            self.visit(stat)

    def visit_Assign(self, node: Assign):
        name = node.left.value
        self.GLOBAL_MEMORY[name] = self.visit(node.right)

    def visit_NoOp(self, node: NoOp):
        pass

    def visit_BinOp(self, node: BinOp):
        if node.op.type == PLUS:
            return self.visit(node.left) + self.visit(node.right)
        elif node.op.type == MINUS:
            return self.visit(node.left) - self.visit(node.right)
        elif node.op.type == MUL:
            return self.visit(node.left) * self.visit(node.right)
        elif node.op.type == INTEGER_DIV:
            return self.visit(node.left) // self.visit(node.right)
        elif node.op.type == FLOAT_DIV:
            return self.visit(node.left) / self.visit(node.right)

    def visit_UnaryOp(self, node: UnaryOp):
        if node.op.type == PLUS:
            return self.visit(node.expr)
        else:   # node.op.type == MINUS
            return -self.visit(node.expr)

    def visit_Var(self, node: Var):
        name = node.value
        val = self.GLOBAL_MEMORY.get(name)
        if val is None:
            raise NameError(repr(name))
        else:
            return val

    def visit_Num(self, node: Num):
        return node.value

    def interpret(self):
        tree = self.tree
        return self.visit(tree)

###############################################################################
#                                                                             #
#   MAIN                                                                      #
#                                                                             #
###############################################################################

if __name__ == '__main__':
    lexer = Lexer(test_code)
    parser = Parser(lexer)
    tree = parser.parse()

    print('')
    print('Abstract Syntax Tree:')
    displayer = Displayer(tree)
    displayer.display()
    print('open "Tree.html"')

    symtab_builder = SymbolTableBuilder()
    symtab_builder.visit(tree)
    print('')
    print('Symbol Table contents:')
    print(symtab_builder.symtab)

    interpreter = Interpreter(tree)
    interpreter.interpret()

    print('')
    print('Run-time GLOBAL_MEMORY contents:')
    for k, v in interpreter.GLOBAL_MEMORY.items():
        print(f'{k} = {v}')