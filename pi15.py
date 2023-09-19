"""
simple pascal interpreter, based on 14.py
- modify LexerError, ParserError, SemanticError, ...
- Token: add position field
- TokenType enum
- reformat code

grammar:
program               : PROGRAM variable SEMI block DOT
block                 : declarations compound_statement
declarations          : (VAR (variable_declaration SEMI)+)? procedure_declaration*
procedure_declaration : PROCEDURE ID (LPAREN formal_params_list RPAREN)? SEMI block SEMI
variable_declaration  : ID (COMMA ID)* COLON type_spec
formal_params_list    : formal_parameters
                      | formal_parameters SEMI formal_params_list
formal_parameters     : ID (COMMA ID)* COLON type_spec
type_spec             : INTEGER | REAL
compound_statement    : BEGIN statement_list END
statement_list        : statement
                      | statement SEMI statement_list
statement             : compound_statement
                      | assignment_statement
                      | empty
assignment_statement  : variable ASSIGN expr
empty                 :
expr                  : term ((PLUS | MINUS) term)*
term                  : factor ((MUL | INTEGER_DIV | FLOAT_DIV) factor)*
factor                : PLUS factor
                      | MINUS factor
                      | INTEGER_CONST
                      | REAL_CONST
                      | LPAREN expr RPAREN
                      | variable
variable              : ID
"""

from collections import OrderedDict, namedtuple
from enum import Enum
import argparse

from pyecharts import options as opts
from pyecharts.charts import Tree
import os

test_code = '''
program Main;
   var
      b, x, y : real;
      z : integer;

   procedure AlphaA(a : integer);
      var b : integer;

      procedure Beta(c : integer);
         var y : integer;

         procedure Gamma(c : integer);
            var x : integer;
         begin { Gamma }
            x := a + b + c + x + y + z;
         end;  { Gamma }

      begin { Beta }

      end;  { Beta }

   begin { AlphaA }

   end;  { AlphaA }

   procedure AlphaB(a : integer);
      var c : real;
   begin { AlphaB }
      c := a + b;
   end;  { AlphaB }

begin { Main }
    x := 0;
    y := x + 1
end.  { Main }
'''

LOCAL_ECHARTS = True
_SHOULD_LOG_SCOPE = False

###############################################################################
#                                                                             #
#   ERROR MESSAGE                                                             #
#                                                                             #
###############################################################################

Position = namedtuple('Position', ['line', 'col'])


class ErrorType(Enum):
    # lexer
    UNRECOGNIZED_CHAR   = 'Unrecognized char'
    # parser, invalid syntax
    UNEXPECTED_TOKEN    = 'Unexpected token'
    # semantic analyser
    ID_NOT_FOUND        = 'Identifier not found'
    DUPLICATE_ID        = 'Duplicate id found'
    # interpreter
    ID_NOT_VISITED      = 'Identifier not visited'


class Error(Exception):
    def __init__(self, position, message):
        self.position = position
        self.message = message

    def __str__(self):
        return f'{self.__class__.__name__}: <{self.position.line}:{self.position.col}>: {self.message}'

    __repr__ = __str__


class LexerError(Error):
    pass


class ParserError(Error):
    pass


class SemanticError(Error):
    pass


class InterpreterError(Error):
    pass


def _gen_error_message(error_code, item):
    if error_code == ErrorType.UNRECOGNIZED_CHAR:
        return f'unrecognized symbol `{item}`'
    elif error_code == ErrorType.UNEXPECTED_TOKEN:
        return f'token of `{item}` is not expected'
    elif error_code == ErrorType.DUPLICATE_ID:
        return f'duplicate identifier `{item}` found'
    elif error_code == ErrorType.ID_NOT_FOUND:
        return f'identifier `{item}` not found'
    elif error_code == ErrorType.ID_NOT_VISITED:
        return f'identifier `{item}` can not visit'


###############################################################################
#                                                                             #
#  LEXER                                                                      #
#                                                                             #
###############################################################################

# Token types
class TokenType(Enum):
    # misc
    ID = 'ID'
    INTEGER_CONST = 'INTEGER_CONST'
    REAL_CONST = 'REAL_CONST'
    EOF = 'EOF'
    # opt
    PLUS = '+'
    MINUS = '-'
    MUL = '*'
    INTEGER_DIV = 'DIV'
    FLOAT_DIV = '/'
    LPAREN = '('
    RPAREN = ')'
    DOT = '.'
    ASSIGN = ':='
    SEMI = ';'
    COMMA = ','
    COLON = ':'
    # keywords
    PROGRAM = 'PROGRAM'
    VAR = 'VAR'
    PROCEDURE = 'PROCEDURE'
    DIV = 'DIV'
    INTEGER = 'INTEGER'
    REAL = 'REAL'
    BEGIN = 'BEGIN'
    END = 'END'


def _build_reserved_keywords():
    tk_list = list(TokenType)
    start_idx = tk_list.index(TokenType.PROGRAM)
    end_idx = tk_list.index(TokenType.END)
    return {
        token_type.value: token_type
        for token_type in tk_list[start_idx: end_idx + 1]
    }


RESERVED_KEYWORDS = _build_reserved_keywords()


class Token:
    def __init__(self, token_type, value, position):
        """Token

        Args:
          token_type: TokenType
          value: str
          position: Position
        """
        self.type = token_type
        self.value = value
        self.position = position

    def __str__(self):
        return f'Token({self.type}, {repr(self.value)}, pos={self.position.line}:{self.position.col})'

    def __repr__(self):
        return self.__str__()


class Lexer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        # self.current_token = None
        self.current_char = self.text[self.pos]
        # for error information
        self.line = 1
        self.col = 0

    def position(self):
        return Position(self.line, self.col)

    def error(self, message):
        raise LexerError(self.position(), message)

    def advance(self):
        """get next char, and increse the pos pointer

        advance the 'pos' pointer and set the 'current_char' variable.
        """
        if self.current_char == '\n':
            self.line += 1
            self.col = 0

        self.pos += 1
        self.col += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None  # end of input
        else:
            self.current_char = self.text[self.pos]

    def peek(self):
        """lookup next char, but not increse the pos pointer
        """
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
        """parse a number from the input
        """
        token = Token(None, None, self.position())

        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()

        if self.current_char != '.':
            token.type = TokenType.INTEGER_CONST
            token.value = int(result)
        else:
            result += self.current_char
            self.advance()

            while self.current_char is not None and self.current_char.isdigit():
                result += self.current_char
                self.advance()

            token.type = TokenType.REAL_CONST
            token.value = float(result)

        return token

    def _id(self):
        """parse an identifier or reserved keyword
        """
        token = Token(None, None, self.position())

        result = ''
        while self.current_char is not None and self.current_char.isalnum():
            result += self.current_char
            self.advance()

        # if keyword, return KW token; else return ID token
        token_type = RESERVED_KEYWORDS.get(result.upper())
        if token_type:
            token.type = token_type
            token.value = result.upper()
        else:
            token.type = TokenType.ID
            token.value = result

        return token

    def get_next_token(self):
        """lexical analyzer, lexer, scanner, tokenizer

        breaking a sentence apart into tokens. One token one time.
        """
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
            if self.current_char == ':' and self.peek() == '=':
                self.advance()
                self.advance()
                return Token(TokenType.ASSIGN, ':=', position=self.position())

            # single-char token
            try:
                token_type = TokenType(self.current_char)
            except ValueError:
                # unrecognized char
                self.error(_gen_error_message(ErrorType.UNRECOGNIZED_CHAR, self.current_char))
            else:
                token = Token(token_type, self.current_char, self.position())
                self.advance()
                return token

        return Token(TokenType.EOF, None, None)


###############################################################################
#                                                                             #
#  AST & PARSER                                                               #
#                                                                             #
###############################################################################

class AST:
    pass


# expr

class Num(AST):
    """
    """

    def __init__(self, token: Token):
        self.token = token
        self.value = token.value


class Var(AST):
    """variable
    """

    def __init__(self, token: Token):
        self.token = token
        self.value = token.value


class UnaryOp(AST):
    """
    """

    def __init__(self, op: Token, expr):
        self.token = self.op = op
        self.expr = expr


class BinOp(AST):
    """
    """

    def __init__(self, left, op: Token, right):
        self.left = left
        self.token = self.op = op
        self.right = right


# stat

class NoOp(AST):
    """empty
    """
    pass


class Assign(AST):
    """assignment_statement
    """

    def __init__(self, left: Var, op: Token, right):
        self.left = left
        self.token = self.op = op
        self.right = right


class Compound(AST):
    """compound_statement & statement_list

    children: statement_list
    """

    def __init__(self):
        self.children = []


class Type(AST):
    """type_spec
    """

    def __init__(self, token: Token):
        self.token = token
        self.value = token.value


class VarDecl(AST):
    """one var decl

    for variable_declaration
    """

    def __init__(self, var_node: Var, type_node: Type):
        self.var_node = var_node
        self.type_node = type_node

    def __str__(self):
        return f'{self.var_node.value, self.type_node.value}'


class Param(AST):
    """one param
    """

    def __init__(self, var_node: Var, type_node: Type):
        self.var_node = var_node
        self.type_node = type_node


class ProcedureDecl(AST):
    """procedure
    """

    def __init__(self, proc_name, params, block_node) -> None:
        """ProcedureDecl

        Args:
          params: list[Param]
          block_node: Block
        """
        self.proc_name = proc_name
        self.params = params
        self.block_node = block_node


class Block(AST):
    """block
    """

    def __init__(self, declarations: list, compound_statement: Compound):
        self.declarations = declarations
        self.compound_statement = compound_statement


class Program(AST):
    """program
    """

    def __init__(self, name, block: Block):
        self.name = name
        self.block = block


class Parser:
    def __init__(self, lexer: Lexer):
        """
        """
        self.lexer = lexer
        self.current_token = self.get_next_token()

    def get_next_token(self):
        return self.lexer.get_next_token()

    def error(self, error_code, token):
        raise ParserError(token.position, _gen_error_message(error_code, token))

    def eat(self, token_type):
        """verify the token type
        """
        if self.current_token.type == token_type:
            self.current_token = self.get_next_token()
        else:
            self.error(ErrorType.UNEXPECTED_TOKEN, self.current_token)

    def program(self):
        """parse program

        program : PROGRAM variable SEMI block DOT
        """
        self.eat(TokenType.PROGRAM)
        prog_name = self.variable().value
        self.eat(TokenType.SEMI)
        block_node = self.block()
        self.eat(TokenType.DOT)
        return Program(prog_name, block_node)

    def block(self):
        """parse block

        block : declarations compound_statement
        """
        declarations_node = self.declarations()
        compound_statement_node = self.compound_statement()
        return Block(declarations_node, compound_statement_node)

    def declarations(self):
        """parse declarations

        declarations : (VAR (variable_declaration SEMI)+)? procedure_declaration*
        """
        declarations_node = []
        if self.current_token.type == TokenType.VAR:
            self.eat(TokenType.VAR)
            while self.current_token.type == TokenType.ID:
                var_decl = self.variable_declaration()
                declarations_node.extend(var_decl)  # add iterable items
                self.eat(TokenType.SEMI)

        while self.current_token.type == TokenType.PROCEDURE:
            proc_decl = self.procedure_declaration()
            declarations_node.append(proc_decl)

        return declarations_node

    def procedure_declaration(self):
        """procedure_declaration

        procedure_declaration : PROCEDURE ID (LPAREN formal_params_list RPAREN)? SEMI block SEMI
        """
        self.eat(TokenType.PROCEDURE)
        proc_name = self.current_token.value
        self.eat(TokenType.ID)

        params = []
        if self.current_token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            params = self.formal_params_list()
            self.eat(TokenType.RPAREN)

        self.eat(TokenType.SEMI)
        block_node = self.block()
        self.eat(TokenType.SEMI)
        proc_decl = ProcedureDecl(proc_name, params, block_node)
        return proc_decl

    def variable_declaration(self):
        """parse variable_declaration

        variable_declaration: ID (COMMA ID)* COLON type_spec
        """
        var_nodes = [Var(self.current_token)]
        self.eat(TokenType.ID)

        while self.current_token.type == TokenType.COMMA:
            self.eat(TokenType.COMMA)
            var_nodes.append(Var(self.current_token))
            self.eat(TokenType.ID)

        self.eat(TokenType.COLON)
        type_node = self.type_spec()
        return [VarDecl(var_node, type_node) for var_node in var_nodes]

    def formal_params_list(self):
        """parse formal_params_list

        formal_params_list : formal_parameters
                           | formal_parameters SEMI formal_params_list
        """
        if not self.current_token.type == TokenType.ID:
            return []

        param_nodes = self.formal_parameters()

        while self.current_token.type == TokenType.SEMI:
            self.eat(TokenType.SEMI)
            param_nodes.extend(self.formal_parameters())

        return param_nodes

    def formal_parameters(self):
        """parse formal_parameters

        formal_parameters : ID (COMMA ID)* COLON type_spec
        """
        param_nodes = []

        param_tokens = [self.current_token]
        self.eat(TokenType.ID)
        while self.current_token.type == TokenType.COMMA:
            self.eat(TokenType.COMMA)
            param_tokens.append(self.current_token)
            self.eat(TokenType.ID)

        self.eat(TokenType.COLON)
        type_node = self.type_spec()

        for token in param_tokens:
            param_nodes.append(Param(Var(token), type_node))

        return param_nodes

    def type_spec(self):
        """parse type_spec

        type_spec : INTEGER | REAL
        """
        token = self.current_token
        if token.type == TokenType.INTEGER:
            self.eat(TokenType.INTEGER)
        else:
            self.eat(TokenType.REAL)
        return Type(token)

    def compound_statement(self):
        """parse compound_statement

        compound_statement : BEGIN statement_list END
        """
        self.eat(TokenType.BEGIN)
        stats = self.statement_list()
        self.eat(TokenType.END)

        result = Compound()
        for stat in stats:
            result.children.append(stat)

        return result

    def statement_list(self):
        """parse statement_list

        statement_list : statement
                       | statement SEMI statement_list
        """
        stat = self.statement()
        result = [stat]

        while self.current_token.type == TokenType.SEMI:
            self.eat(TokenType.SEMI)
            result.append(self.statement())

        return result

    def statement(self):
        """parse statement

        statement : compound_statement
                  | assignment_statement
                  | empty
        """
        if self.current_token.type == TokenType.BEGIN:
            result = self.compound_statement()
        elif self.current_token.type == TokenType.ID:
            result = self.assignment_statement()
        else:
            result = self.empty()
        return result

    def assignment_statement(self):
        """parse assignment_statement

        assignment_statement : variable ASSIGN expr
        """
        left = self.variable()
        token = self.current_token
        self.eat(TokenType.ASSIGN)
        right = self.expr()

        result = Assign(left, token, right)
        return result

    def empty(self):
        """parse empty
        """
        return NoOp()

    def expr(self):
        """parse expr

        expr: term ((PLUS | MINUS) term)*
        """
        result = self.term()

        while self.current_token.type in (TokenType.PLUS, TokenType.MINUS):
            op = self.current_token
            if op.type == TokenType.PLUS:
                self.eat(TokenType.PLUS)
            else:
                self.eat(TokenType.MINUS)
            result = BinOp(left=result, op=op, right=self.term())

        return result

    def term(self):
        """parse term

        term: factor ((MUL | INTEGER_DIV | FLOAT_DIV) factor)*
        """
        result = self.factor()

        while self.current_token.type in (TokenType.MUL, TokenType.INTEGER_DIV, TokenType.FLOAT_DIV):
            op = self.current_token
            if op.type == TokenType.MUL:
                self.eat(TokenType.MUL)
            elif op.type == TokenType.INTEGER_DIV:
                self.eat(TokenType.INTEGER_DIV)
            elif op.type == TokenType.FLOAT_DIV:
                self.eat(TokenType.FLOAT_DIV)
            result = BinOp(left=result, op=op, right=self.factor())

        return result

    def factor(self):
        """parse factor

        factor : PLUS factor
               | MINUS factor
               | INTEGER_CONST
               | REAL_CONST
               | LPAREN expr RPAREN
               | variable
        """
        token = self.current_token
        if token.type == TokenType.PLUS:
            self.eat(TokenType.PLUS)
            result = UnaryOp(token, self.factor())
            return result
        elif token.type == TokenType.MINUS:
            self.eat(TokenType.MINUS)
            result = UnaryOp(token, self.factor())
            return result
        elif token.type == TokenType.INTEGER_CONST:
            self.eat(TokenType.INTEGER_CONST)
            return Num(token)
        elif token.type == TokenType.REAL_CONST:
            self.eat(TokenType.REAL_CONST)
            return Num(token)
        elif token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            result = self.expr()
            self.eat(TokenType.RPAREN)
            return result
        else:
            return self.variable()

    def variable(self):
        """parse variable
        """
        result = Var(self.current_token)
        self.eat(TokenType.ID)
        return result

    def parse(self):
        result = self.program()
        if self.current_token.type != TokenType.EOF:
            self.error(ErrorType.UNEXPECTED_TOKEN, self.current_token)
        return result


###############################################################################
#                                                                             #
#  NodeVistor                                                                 #
#                                                                             #
###############################################################################

class NodeVistor:
    def visit(self, node):
        """dispatches
        """
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
        self.tree = tree

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
        params = {
            'name': f'Params',
            'children': []
        }
        for param in node.params:
            params['children'].append(self.visit(param))
        data = {
            'name': f'Proc-{node.proc_name}',
            'children': [params, self.visit(node.block_node)]
        }
        return data

    def visit_Param(self, node: Param):
        data = {
            'name': f'{node.var_node.value}',
            'children': [
                {'name': f'{node.type_node.value}'},
            ]
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
        data = self.visit(self.tree)
        c = (
            Tree()
            .add(
                series_name="",  # name
                data=[data],  # data
                initial_tree_depth=-1,  # all expand
                orient="TB",  # top-to-bottom
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
#  SYMBOL & SEMANTIC ANALYZER                                                 #
#                                                                             #
###############################################################################

class Symbol:
    def __init__(self, name, type=None):
        self.name = name
        self.type = type


class BuiltinTypeSymbol(Symbol):
    def __init__(self, name):
        super().__init__(name)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}(name="{self.name}")>'


class VarSymbol(Symbol):
    def __init__(self, name, type: BuiltinTypeSymbol):
        """VarSymbol

        Args:
          name: str
          type: BuiltinTypeSymbol
        """
        super().__init__(name, type)

    def __str__(self) -> str:
        return "<{class_name}(name='{name}', type='{type}')>".format(
            class_name=self.__class__.__name__,
            name=self.name,
            type=self.type,
        )

    __repr__ = __str__


class ProcedureSymbol(Symbol):
    def __init__(self, name, params):
        """ProcedureSymbol

        Args:
          name: name
          params: list[VarSymbol]
        """
        super().__init__(name)  # ProcedureSymbol don't need type field
        self.params = params

    def __str__(self) -> str:
        return "<{class_name}(name='{name}', parameters='{params}')>".format(
            class_name=self.__class__.__name__,
            name=self.name,
            params=self.params,
        )

    __repr__ = __str__


class ScopedSymbolTable:
    def __init__(self, scope_name, scope_level, enclosing_scope):
        self._symbols = OrderedDict()
        self.scope_name = scope_name
        self.scope_level = scope_level
        self.enclosing_scope = enclosing_scope

    def __str__(self) -> str:
        h1 = 'SCOPE (SCOPED SYMBOL TABLE)'
        lines = ['', h1, '=' * len(h1)]
        for header_name, header_value in (
            ('Scope name', self.scope_name),
            ('Scope level', self.scope_level),
            ('Enclosing scope', self.enclosing_scope.scope_name if self.enclosing_scope else None)
        ):
            lines.append('%-15s: %s' % (header_name, header_value))
        h2 = 'Scope (Scoped symbol table) contents'
        lines.extend([h2, '-' * len(h2)])
        # symtab_header = 'Symbol table contents:'
        # lines = [symtab_header, '_' * len(symtab_header)]
        lines.extend(
            ['%7s: %r' % (key, value) for key, value in self._symbols.items()]
        )
        s = '\n'.join(lines)
        return s

    __repr__ = __str__

    def init_builtins(self):
        self.insert(BuiltinTypeSymbol('INTEGER'))
        self.insert(BuiltinTypeSymbol('REAL'))

    def log(self, msg):
        if _SHOULD_LOG_SCOPE:
            print(msg)

    def insert(self, symbol: Symbol):
        self.log(f'insert: {symbol.name}. (scope name: {self.scope_name})')
        self._symbols[symbol.name] = symbol

    def lookup(self, name, current_scope_only=False) -> Symbol:
        """lookup a Symbol

        if not found, return None
        """
        self.log(f'lookup: {name}. (scope name: {self.scope_name})')
        symbol = self._symbols.get(name)

        if current_scope_only or symbol is not None:
            return symbol

        # recursively lookup symbol in the outer scope
        if self.enclosing_scope is not None:
            return self.enclosing_scope.lookup(name)


class SemanticAnalyzer(NodeVistor):
    def __init__(self, tree):
        self.current_scope = None
        self.tree = tree

    def error(self, error_code, token):
        raise SemanticError(token.position, _gen_error_message(error_code, token))

    def log(self, msg):
        if _SHOULD_LOG_SCOPE:
            print(msg)

    def visit_Program(self, node: Program):
        self.log('enter scope: global')
        global_scope = ScopedSymbolTable(
            scope_name='global',
            scope_level=self.current_scope.scope_level + 1,
            enclosing_scope=self.current_scope)
        # global_scope._init_builtins()
        self.current_scope = global_scope
        # visit subtree
        self.visit(node.block)
        self.log(global_scope)
        # leave
        self.current_scope = self.current_scope.enclosing_scope
        self.log('leave scope: global')

    def visit_Block(self, node: Block):
        for declaration in node.declarations:
            self.visit(declaration)
        self.visit(node.compound_statement)

    def visit_ProcedureDecl(self, node: ProcedureDecl):
        proc_name = node.proc_name
        proc_symbol = ProcedureSymbol(name=proc_name, params=[])  # params is now empty
        # add procedure symbol in current scope
        self.current_scope.insert(proc_symbol)

        # enter scope of procedure
        self.log(f'enter scope: {proc_name}')
        proc_scope = ScopedSymbolTable(
            scope_name=proc_name,
            scope_level=self.current_scope.scope_level + 1,
            enclosing_scope=self.current_scope)
        self.current_scope = proc_scope

        # add param symbols
        for param in node.params:
            param_type = self.current_scope.lookup(param.type_node.value)
            param_symbol = VarSymbol(param.var_node.value, param_type)
            self.current_scope.insert(param_symbol)
            proc_symbol.params.append(param_symbol)  # add param symbol to proc_symbol.params

        self.visit(node.block_node)

        self.log(proc_scope)

        # leave
        self.current_scope = self.current_scope.enclosing_scope
        self.log('leave scope: %s' % proc_name)

    def visit_Param(self, node: Param):
        pass

    def visit_VarDecl(self, node: VarDecl):
        # TODO: make sure type_symbol is really a type
        type_symbol = self.current_scope.lookup(node.type_node.value)
        var_name = node.var_node.value

        # current scope has defined a symbol with this name
        if self.current_scope.lookup(var_name, current_scope_only=True) is not None:
            self.error(ErrorType.DUPLICATE_ID, node.var_node.token)

        var_symbol = VarSymbol(var_name, type_symbol)
        self.current_scope.insert(var_symbol)

    def visit_Type(self, node: Type):
        pass

    def visit_Compound(self, node: Compound):
        for stat in node.children:
            self.visit(stat)

    def visit_Assign(self, node: Assign):
        """check if left var has defined
        """
        var_name = node.left.value
        var_sybl = self.current_scope.lookup(var_name)
        if var_sybl is None:
            self.error(ErrorType.ID_NOT_FOUND, node.left.token)
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
        """check var has defined
        """
        var_name = node.value
        var_sybl = self.current_scope.lookup(var_name)
        if var_sybl is None:
            self.error(ErrorType.ID_NOT_FOUND, node.token)

    def visit_Num(self, node: Num):
        pass

    def analysis(self):
        scope_0_name = '__scope_0'
        self.log(f'enter scope: {scope_0_name}')
        scope_0 = ScopedSymbolTable(
            scope_name=scope_0_name,
            scope_level=0,
            enclosing_scope=None)
        scope_0.init_builtins()
        self.current_scope = scope_0

        self.visit(self.tree)
        self.log(scope_0)
        self.log(f'leave scope: {scope_0_name}')


###############################################################################
#                                                                             #
#   INTERPRETER                                                               #
#                                                                             #
###############################################################################

class Interpreter(NodeVistor):
    def __init__(self, tree) -> None:
        self.tree = tree
        self.GLOBAL_MEMORY = {}  # symbol table

    def error(self, error_code, token):
        raise InterpreterError(token.position, _gen_error_message(error_code, token))

    def visit_Program(self, node: Program):
        self.visit(node.block)

    def visit_Block(self, node: Block):
        for declaration in node.declarations:
            self.visit(declaration)
        self.visit(node.compound_statement)

    def visit_ProcedureDecl(self, node: ProcedureDecl):
        pass

    def visit_Param(self, node: Param):
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
        if node.op.type == TokenType.PLUS:
            return self.visit(node.left) + self.visit(node.right)
        elif node.op.type == TokenType.MINUS:
            return self.visit(node.left) - self.visit(node.right)
        elif node.op.type == TokenType.MUL:
            return self.visit(node.left) * self.visit(node.right)
        elif node.op.type == TokenType.INTEGER_DIV:
            return self.visit(node.left) // self.visit(node.right)
        elif node.op.type == TokenType.FLOAT_DIV:
            return self.visit(node.left) / self.visit(node.right)

    def visit_UnaryOp(self, node: UnaryOp):
        if node.op.type == TokenType.PLUS:
            return self.visit(node.expr)
        else:  # node.op.type == MINUS
            return -self.visit(node.expr)

    def visit_Var(self, node: Var):
        name = node.value
        val = self.GLOBAL_MEMORY.get(name)
        if val is None:
            self.error(ErrorType.ID_NOT_VISITED, node.token)
        else:
            return val

    def visit_Num(self, node: Num):
        return node.value

    def interpret(self):
        return self.visit(self.tree)


###############################################################################
#                                                                             #
#   MAIN                                                                      #
#                                                                             #
###############################################################################

def spi_main():
    global _SHOULD_LOG_SCOPE
    pass


if __name__ == '__main__':
    # global _SHOULD_LOG_SCOPE

    parser = argparse.ArgumentParser(description='SPI - Simple Pascal Interpreter')
    # parser.add_argument('inputfile', help='Pascal source file')
    parser.add_argument(
        '--scope',
        help='Print scope information',
        action='store_true',
    )
    args = parser.parse_args()

    _SHOULD_LOG_SCOPE = args.scope

    try:
        # spi_main()
        lexer = Lexer(test_code)
        parser = Parser(lexer)
        tree = parser.parse()

        print('')
        print('Abstract Syntax Tree:')
        displayer = Displayer(tree)
        displayer.display()
        print('open "Tree.html"')
        print('')

        semantic_analyzer = SemanticAnalyzer(tree)
        semantic_analyzer.analysis()

        interpreter = Interpreter(tree)
        interpreter.interpret()

        print('')
        print('Run-time GLOBAL_MEMORY contents:')
        for k, v in interpreter.GLOBAL_MEMORY.items():
            print(f'{k} = {v}')
    except (LexerError, ParserError, SemanticError, InterpreterError) as e:
        print(e)
