"""
simple pascal interpreter, based on 21.py
- string
- build-in function
- modify call strategy

grammar:
program               : PROGRAM variable SEMI block DOT
block                 : declarations compound_statement
declarations          : ((VAR (variable_declaration SEMI)+)? | procedure_declaration | function_declaration)*
procedure_declaration : PROCEDURE ID (LPAREN formal_params_list RPAREN)? SEMI block SEMI
function_declaration  : FUNCTION ID (LPAREN formal_params_list RPAREN)? COLON type_spec SEMI block SEMI
variable_declaration  : ID (COMMA ID)* COLON type_spec
formal_params_list    : formal_parameters
                      | formal_parameters SEMI formal_params_list
formal_parameters     : ID (COMMA ID)* COLON type_spec
type_spec             : INTEGER | REAL | STRING
compound_statement    : BEGIN statement_list END
statement_list        : statement
                      | statement SEMI statement_list
statement             : compound_statement
                      | call
                      | assignment_statement
                      | if_statement
                      | while_statement
                      | empty
call                  : ID LPAREN (expr (COMMA expr)*)? RPAREN
assignment_statement  : variable ASSIGN expr
if_statement          : IF expr THEN statement (ELSE statement)?
while_statement       : WHILE expr DO statement
empty                 :
expr                  : addition_expr ((EQUAL | NOT_EQUAL | GREATER | LESS) addition_expr)?
addition_expr         : multiple_expr ((PLUS | MINUS) multiple_expr)*
multiple_expr         : factor ((MUL | INTEGER_DIV | FLOAT_DIV) factor)*
factor                : PLUS factor
                      | MINUS factor
                      | INTEGER_CONST
                      | REAL_CONST
                      | STRING_LITERAL
                      | LPAREN expr RPAREN
                      | variable
                      | call
variable              : ID
"""

from collections import OrderedDict, namedtuple
from enum import Enum
import argparse

from pyecharts import options as opts
from pyecharts.charts import Tree
import os

test_code = '''
program factorial;

function factorial(n: integer): integer;
begin
    if n = 0 then
        factorial := 1
    else
        factorial := n * factorial(n - 1);
end;

function fibonacci(n: integer): integer;
begin
    if n = 0 then
        fibonacci := 1
    else if n = 1 then
        fibonacci := 1
    else
        fibonacci := fibonacci(n - 1) + fibonacci(n - 2);
end;

var
    n: integer;

begin
    writeln('factorial:');
    n := 0;
    while n < 10 do
    begin
        writeln(' ', n, ': ', factorial(n));
        n := n + 1
    end;

    writeln('fibonacci:');
    n := 0;
    while n < 10 do
    begin
        writeln(' ', n, ': ', fibonacci(n));
        n := n + 1
    end
end.
'''

LOCAL_ECHARTS = True
_SHOULD_LOG_SCOPE = False
_SHOULD_LOG_STACK = False

BUILDIN_FUNCTIONS = {
    'writeln': print
}

###############################################################################
#                                                                             #
#   ERROR MESSAGE                                                             #
#                                                                             #
###############################################################################

Position = namedtuple('Position', ['line', 'col'])


class ErrorInfo:
    # lexer error

    @staticmethod
    def unrecognized_char(item):
        return f'unrecognized char `{item}`'

    @staticmethod
    def literal_string_not_end():
        return f'literal string is not end'

    # parser error

    @staticmethod
    def unexpected_token(item, want):
        return f'token `{item}` is not expected, want `{want}`'

    @staticmethod
    def unsupported_type(item):     # info: code not completed
        return f'type `{item}` is not supported'

    # semantic error

    @staticmethod
    def id_duplicate_defined(item):
        return f'identifier `{item}` duplicate defined'

    @staticmethod
    def id_not_defined(item):
        return f'identifier `{item}` not defined'

    @staticmethod
    def id_not_callable(item):
        return f'identifier `{item}` not a procedure/function'

    @staticmethod
    def wrong_arguments_num(item):
        return f'wrong argument number in procedure `{item}` call'

    # intepreter error

    @staticmethod
    def id_not_valid(item):
        return f'identifier `{item}` can not visit'

    @staticmethod
    def unsupported_op(item):        # info: code not completed
        return f'op `{item}` is not supported'


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


###############################################################################
#                                                                             #
#  LEXER                                                                      #
#                                                                             #
###############################################################################

# Token types
class TokenType(Enum):
    # misc
    ID              = 'ID'
    INTEGER_CONST   = 'INTEGER_CONST'
    REAL_CONST      = 'REAL_CONST'
    STRING_LITERAL  = 'STRING_LITERAL'
    EOF             = 'EOF'
    # opt
    PLUS            = '+'
    MINUS           = '-'
    MUL             = '*'
    INTEGER_DIV     = 'DIV'
    FLOAT_DIV       = '/'
    LPAREN          = '('
    RPAREN          = ')'
    DOT             = '.'
    ASSIGN          = ':='
    SEMI            = ';'
    COMMA           = ','
    COLON           = ':'
    EQUAL           = '='
    NOT_EQUAL       = '<>'
    GREATER         = '>'
    LESS            = '<'
    # keywords
    PROGRAM         = 'PROGRAM'
    VAR             = 'VAR'
    PROCEDURE       = 'PROCEDURE'
    FUNCTION        = 'FUNCTION'
    DIV             = 'DIV'
    INTEGER         = 'INTEGER'
    REAL            = 'REAL'
    STRING          = 'STRING'
    IF              = 'IF'
    THEN            = 'THEN'
    ELSE            = 'ELSE'
    WHILE           = 'WHILE'
    DO              = 'DO'
    BEGIN           = 'BEGIN'
    END             = 'END'


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

    def string_literal(self):
        """parse a literal string like '***'
        """
        token = Token(TokenType.STRING_LITERAL, None, self.position())
        result = ''
        self.advance()
        while self.current_char != '\'':
            if self.current_char is None:
                self.error(ErrorInfo.literal_string_not_end())
            result += self.current_char
            self.advance()
        self.advance()
        token.value = result
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

            # <>
            if self.current_char == '<' and self.peek() == '>':
                self.advance()
                self.advance()
                return Token(TokenType.NOT_EQUAL, '<>', position=self.position())

            # '
            if self.current_char == '\'':
                return self.string_literal()

            # single-char token
            try:
                token_type = TokenType(self.current_char)
            except ValueError:
                # unrecognized char
                self.error(ErrorInfo.unrecognized_char(self.current_char))
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


class String(AST):
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

class While(AST):
    """while_statement
    """
    def __init__(self, token, cond_expr, do_stat):
        '''if

        Args:
          token: Token `while`
          cond_expr: expr
          do_stat: stat
        '''
        self.token = token
        self.cond_expr = cond_expr
        self.do_stat = do_stat


class If(AST):
    """if_statement
    """
    def __init__(self, token, cond_expr, then_stat, else_stat):
        '''if

        Args:
          token: Token `if`
          cond_expr: expr
          then_stat: stat
          else_stat: stat
        '''
        self.token = token
        self.cond_expr = cond_expr
        self.then_stat = then_stat
        self.else_stat = else_stat


class Assign(AST):
    """assignment_statement
    """

    def __init__(self, left: Var, op: Token, right):
        self.left = left
        self.token = self.op = op
        self.right = right


class Call(AST):
    """call
    """

    def __init__(self, name: str, arguments: list, token: Token):
        self.name = name
        self.arguments = arguments
        self.token = token
        self.call_symbol = None   # set it in SemanticAnalyzer


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


class Param(AST):
    """one param
    """

    def __init__(self, var_node: Var, type_node: Type):
        self.var_node = var_node
        self.type_node = type_node


class VarDecl(AST):
    """one var decl

    for variable_declaration
    """

    def __init__(self, var_node: Var, type_node: Type):
        self.var_node = var_node
        self.type_node = type_node

    def __str__(self):
        return f'{self.var_node.value, self.type_node.value}'


class FunctionDecl(AST):
    """function
    """

    def __init__(self, func_name, params, block_node, type_node):
        """FunctionDecl

        Args:
          params: list[Param]
          block_node: Block
          type_node: Type
        """
        self.func_name = func_name
        self.params = params
        self.block_node = block_node
        self.type_node = type_node

    def __str__(self):
        params = [f'{p.var_node.value}: {p.type_node.value}' for p in self.params]
        return f'{self.func_name}({", ".join(params)})'


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

    def __str__(self):
        params = [f'{p.var_node.value}: {p.type_node.value}' for p in self.params]
        return f'{self.proc_name}({", ".join(params)})'


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

    def error(self, token, message):
        raise ParserError(token.position, message)

    def eat(self, token_type):
        """verify the token type
        """
        if self.current_token.type == token_type:
            self.current_token = self.get_next_token()
        else:
            self.error(self.current_token, ErrorInfo.unexpected_token(self.current_token, token_type))

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

        declarations : ((VAR (variable_declaration SEMI)+)? | procedure_declaration | function_declaration)*
        """
        declarations_node = []

        while self.current_token.type in (TokenType.PROCEDURE, TokenType.FUNCTION, TokenType.VAR):
            if self.current_token.type == TokenType.PROCEDURE:
                decl = self.procedure_declaration()
                declarations_node.append(decl)
            elif self.current_token.type == TokenType.FUNCTION:
                decl = self.function_declaration()
                declarations_node.append(decl)
            else:
                self.eat(TokenType.VAR)
                while self.current_token.type == TokenType.ID:
                    var_decl = self.variable_declaration()
                    declarations_node.extend(var_decl)  # add iterable items
                    self.eat(TokenType.SEMI)

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

    def function_declaration(self):
        """function_declaration

        function_declaration : FUNCTION ID (LPAREN formal_params_list RPAREN)? COLON type_spec SEMI block SEMI
        """
        self.eat(TokenType.FUNCTION)
        func_name = self.current_token.value
        self.eat(TokenType.ID)

        params = []
        if self.current_token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            params = self.formal_params_list()
            self.eat(TokenType.RPAREN)

        self.eat(TokenType.COLON)
        type_node = self.type_spec()

        self.eat(TokenType.SEMI)
        block_node = self.block()
        self.eat(TokenType.SEMI)
        proc_decl = FunctionDecl(func_name, params, block_node, type_node)
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

        type_spec : INTEGER | REAL | STRING
        """
        token = self.current_token
        if token.type in (TokenType.INTEGER, TokenType.REAL, TokenType.STRING):
            self.eat(token.type)
            return Type(token)
        else:
            self.error(token, ErrorInfo.unsupported_type(token.value))

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
                  | call
                  | assignment_statement
                  | if_statement
                  | while_statement
                  | empty
        """
        if self.current_token.type == TokenType.BEGIN:
            result = self.compound_statement()
        elif self.current_token.type == TokenType.ID:
            if self.lexer.current_char == '(':
                result = self.call()
            else:
                result = self.assignment_statement()
        elif self.current_token.type == TokenType.IF:
            result = self.if_statement()
        elif self.current_token.type == TokenType.WHILE:
            result = self.while_statement()
        else:
            result = self.empty()
        return result

    def call(self):
        """parse call

        call : ID LPAREN (expr (COMMA expr)*)? RPAREN
        """
        token = self.current_token
        name = token.value
        self.eat(TokenType.ID)
        self.eat(TokenType.LPAREN)

        arguments = []
        if self.current_token.type != TokenType.RPAREN:
            arguments.append(self.expr())

            while self.current_token.type == TokenType.COMMA:
                self.eat(TokenType.COMMA)
                arguments.append(self.expr())

        self.eat(TokenType.RPAREN)

        return Call(name, arguments, token)

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

    def if_statement(self):
        """parse if_statement

        if_statement : IF expr THEN statement (ELSE statement)?
        """
        token = self.current_token
        self.eat(TokenType.IF)
        cond_expr = self.expr()
        self.eat(TokenType.THEN)
        then_stat = self.statement()
        else_stat = None
        if self.current_token.type == TokenType.ELSE:
            self.eat(TokenType.ELSE)
            else_stat = self.statement()

        return If(token, cond_expr, then_stat, else_stat)

    def while_statement(self):
        """parse while_statement

        while_statement : WHILE expr DO statement
        """
        token = self.current_token
        self.eat(TokenType.WHILE)
        cond_expr = self.expr()
        self.eat(TokenType.DO)
        do_stat = self.statement()

        return While(token, cond_expr, do_stat)

    def empty(self):
        """parse empty
        """
        return NoOp()

    def expr(self):
        """parse expr

        expr : addition_expr ((EQUAL | NOT_EQUAL | GREATER | LESS) addition_expr)?
        """
        result = self.addition_expr()

        if self.current_token.type in (TokenType.EQUAL, TokenType.NOT_EQUAL, TokenType.GREATER, TokenType.LESS):
            op = self.current_token
            self.eat(self.current_token.type)
            result = BinOp(left=result, op=op, right=self.addition_expr())

        return result

    def addition_expr(self):
        """parse addition_expr

        addition_expr : multiple_expr ((PLUS | MINUS) multiple_expr)*
        """
        result = self.multiple_expr()

        while self.current_token.type in (TokenType.PLUS, TokenType.MINUS):
            op = self.current_token
            if op.type == TokenType.PLUS:
                self.eat(TokenType.PLUS)
            else:
                self.eat(TokenType.MINUS)
            result = BinOp(left=result, op=op, right=self.multiple_expr())

        return result

    def multiple_expr(self):
        """parse multiple_expr

        multiple_expr : factor ((MUL | INTEGER_DIV | FLOAT_DIV) factor)*
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
               | STRING_LITERAL
               | LPAREN expr RPAREN
               | variable
               | call
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
        elif token.type == TokenType.STRING_LITERAL:
            self.eat(TokenType.STRING_LITERAL)
            return String(token)
        elif token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            result = self.expr()
            self.eat(TokenType.RPAREN)
            return result
        elif token.type == TokenType.ID:
            if self.lexer.current_char == '(':
                return self.call()
            else:
                return self.variable()
        else:
            self.error(token, ErrorInfo.unexpected_token(token, None))

    def variable(self):
        """parse variable
        """
        result = Var(self.current_token)
        self.eat(TokenType.ID)
        return result

    def parse(self):
        result = self.program()
        if self.current_token.type != TokenType.EOF:
            self.error(self.current_token, ErrorInfo.unexpected_token(self.current_token, TokenType.EOF))

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
            'name': f'Proc: {node.proc_name}',
            'children': []
        }
        if len(node.params) != 0:
            data['children'].append(params)
        data['children'].append(self.visit(node.block_node))
        return data

    def visit_FunctionDecl(self, node: FunctionDecl):
        params = {
            'name': f'Params',
            'children': []
        }
        for param in node.params:
            params['children'].append(self.visit(param))
        data = {
            'name': f'Proc: {node.func_name}',
            'children': []
        }
        if len(node.params) != 0:
            data['children'].append(params)
        data['children'].append(self.visit(node.block_node))
        data['children'].append(self.visit(node.type_node))
        return data

    def visit_VarDecl(self, node: VarDecl):
        data = {
            'name': f'Var: {node.var_node.value}',
            'children': [self.visit(node.type_node)]
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

    def visit_Call(self, node: Call):
        data = {
            'name': f'Call:{node.name}',
            'children': []
        }
        for arg in node.arguments:
            data['children'].append(self.visit(arg))
        return data

    def visit_Assign(self, node: Assign):
        data = {
            'name': ':=',
            'children': []
        }
        data['children'].append(self.visit(node.left))
        data['children'].append(self.visit(node.right))
        return data

    def visit_If(self, node: If):
        data = {
            'name': 'if',
            'children': []
        }
        data['children'].append({'name': 'cond', 'children': [self.visit(node.cond_expr)]})
        data['children'].append({'name': 'then', 'children': [self.visit(node.then_stat)]})
        if node.else_stat is not None:
            data['children'].append({'name': 'else', 'children': [self.visit(node.else_stat)]})
        return data

    def visit_While(self, node: While):
        data = {
            'name': 'while',
            'children': []
        }
        data['children'].append({'name': 'cond', 'children': [self.visit(node.cond_expr)]})
        data['children'].append({'name': 'do', 'children': [self.visit(node.do_stat)]})
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

    def visit_String(self, node: String):
        data = {
            'name': f'{str(node.value)}'
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
        self.scope_level = 0


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
    def __init__(self, name, params, block_ast):
        """ProcedureSymbol

        Args:
          name: name
          params: list[VarSymbol]
        """
        super().__init__(name)  # ProcedureSymbol don't need type field
        self.params = params
        self.block_ast = block_ast

    def __str__(self) -> str:
        return "<{class_name}(name='{name}', parameters='{params}')>".format(
            class_name=self.__class__.__name__,
            name=self.name,
            params=self.params,
        )

    __repr__ = __str__


class FunctionSymbol(Symbol):
    def __init__(self, name, params, type, block_ast):
        """FunctionSymbol

        Args:
          name: name
          params: list[VarSymbol]
          type: BuiltinTypeSymbol
        """
        super().__init__(name, type)  # FunctionSymbol has return type
        self.params = params
        self.block_ast = block_ast

    def __str__(self) -> str:
        return "<{class_name}(name='{name}', parameters='{params}', return='{type}')>".format(
            class_name=self.__class__.__name__,
            name=self.name,
            params=self.params,
            type=self.type
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
        self.insert(BuiltinTypeSymbol('STRING'))

    def log(self, msg):
        if _SHOULD_LOG_SCOPE:
            print(msg)

    def insert(self, symbol: Symbol):
        self.log(f'insert: {symbol.name}. (scope name: {self.scope_name})')
        symbol.scope_level = self.scope_level
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
        else:
            return None

class SemanticAnalyzer(NodeVistor):
    def __init__(self, tree):
        self.current_scope = None
        self.tree = tree

    def error(self, token, message):
        raise SemanticError(token.position, message)

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
        proc_symbol = ProcedureSymbol(
            name=proc_name,
            params=[],
            block_ast=node.block_node
        )
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

    def visit_FunctionDecl(self, node: FunctionDecl):
        func_name = node.func_name
        func_symbol = FunctionSymbol(
            name=func_name,
            params=[],
            type=self.current_scope.lookup(node.type_node.value),
            block_ast=node.block_node
        )
        # add function symbol in current scope
        self.current_scope.insert(func_symbol)

        # enter scope of function
        self.log(f'enter scope: {func_name}')
        func_scope = ScopedSymbolTable(
            scope_name=func_name,
            scope_level=self.current_scope.scope_level + 1,
            enclosing_scope=self.current_scope)
        self.current_scope = func_scope

        # add param symbols
        for param in node.params:
            param_type = self.current_scope.lookup(param.type_node.value)
            param_symbol = VarSymbol(param.var_node.value, param_type)
            self.current_scope.insert(param_symbol)
            func_symbol.params.append(param_symbol)  # add param symbol to proc_symbol.params

        self.visit(node.block_node)

        self.log(func_scope)

        # leave
        self.current_scope = self.current_scope.enclosing_scope
        self.log('leave scope: %s' % func_name)

    def visit_VarDecl(self, node: VarDecl):
        # TODO: make sure type_symbol is really a type
        type_symbol = self.current_scope.lookup(node.type_node.value)
        var_name = node.var_node.value

        # current scope has defined a symbol with this name
        if self.current_scope.lookup(var_name, current_scope_only=True) is not None:
            self.error(node.var_node.token, ErrorInfo.id_duplicate_defined(node.var_node.token))

        var_symbol = VarSymbol(var_name, type_symbol)
        self.current_scope.insert(var_symbol)

    def visit_Param(self, node: Param):
        pass

    def visit_Type(self, node: Type):
        pass

    def visit_Compound(self, node: Compound):
        for stat in node.children:
            self.visit(stat)

    def visit_Call(self, node: Call):
        # check procedure call name
        call_symbol = self.current_scope.lookup(node.name)
        if call_symbol is None and node.name in BUILDIN_FUNCTIONS:        # buildin functions
            # print(f'[!] try to call buildin function `{node.name}`')
            pass
        elif type(call_symbol) in (ProcedureSymbol, FunctionSymbol):      # proc or func
            # check param num
            if len(call_symbol.params) != len(node.arguments):
                self.error(node.token, ErrorInfo.wrong_arguments_num(node.token))

            # NOTICE: importent
            node.call_symbol = call_symbol
        else:
            self.error(node.token, ErrorInfo.id_not_callable(node.token))

        for arg_node in node.arguments:
            self.visit(arg_node)

    def visit_Assign(self, node: Assign):
        """check if left var has defined
        """
        var_name = node.left.value
        var_sybl = self.current_scope.lookup(var_name)
        if var_sybl is None:
            self.error(node.left.token, ErrorInfo.id_not_defined(node.left.token))
        else:
            self.visit(node.right)

    def visit_If(self, node: If):
        self.visit(node.cond_expr)
        self.visit(node.then_stat)
        if node.else_stat is not None:
            self.visit(node.else_stat)

    def visit_While(self, node: While):
        self.visit(node.cond_expr)
        self.visit(node.do_stat)

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
            self.error(node.token, ErrorInfo.id_not_defined(node.token))

    def visit_String(self, node: String):
        pass

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

class ARType(Enum):
    PROGRAM   = 'PROGRAM'
    PROCEDURE = 'PROCEDURE'

class ActivationRecord:
    def __init__(self, name, type: ARType):
        self.name           = name
        self.type           = type
        self.members        = {}
        self.enclosing      = None  # outer ActivationRecord
        self.nesting_level  = 0

    def __setitem__(self, key, value):
        self.members[key] = value

    def __getitem__(self, key):
        return self.members[key]

    def get(self, key):
        return self.members.get(key)

    def get_value(self, key, current_level_only=False):
        value = self.members.get(key)

        if value is not None or current_level_only:
            return value

        if self.enclosing is not None:
            return self.enclosing.get_value(key)
        else:
            return None

    def set_value(self, key, value, current_level_only=False):
        if key in self.members or current_level_only:
            self.members[key] = value
            return

        if self.enclosing is not None:
            self.enclosing.set_value(key, value)

    def get_enclosing(self):
        return self.enclosing

    def set_enclosing(self, enclosing):
        self.enclosing = enclosing

    def set_nesting_level(self, level):
        self.nesting_level = level

    def has(self, key):
        return key in self.members

    def __str__(self) -> str:
        lines = [
            '{level}: {type} {name}'.format(
                level=self.nesting_level,
                type=self.type.value,
                name=self.name
            )
        ]
        for name, val in self.members.items():
            lines.append(f'    {name:<20}: {val}')
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return self.__str__()


class CallStack:
    def __init__(self) -> None:
        self._records = []
        self.current = None
        self.current_level = 0

    def push(self, ar: ActivationRecord):
        ar.set_enclosing(self.current)
        ar.set_nesting_level(self.current_level+1)
        self._records.append(ar)
        self.current_level += 1
        self.current = ar

    def pop(self) -> ActivationRecord:
        self.current_level -= 1
        self.current = self.current.get_enclosing()
        return self._records.pop()

    def peek(self) -> ActivationRecord:
        return self._records[-1]

    def __str__(self) -> str:
        s = '\n'.join(repr(ar) for ar in reversed(self._records))
        return f'CALL STACK\n{s}\n'


class Interpreter(NodeVistor):
    def __init__(self, tree) -> None:
        self.tree = tree
        self.call_stack = CallStack()

    def error(self, token, message):
        raise InterpreterError(token.position, message)

    def log(self, msg):
        if _SHOULD_LOG_STACK:
            print(msg)

    def visit_Program(self, node: Program):
        prog_name = node.name

        ar = ActivationRecord(prog_name, ARType.PROGRAM)

        self.call_stack.push(ar)

        self.log(f'enter: PROGRAM {prog_name}')
        self.log(self.call_stack)

        self.visit(node.block)

        self.log(f'leave: PROGRAM {prog_name}')
        self.log(self.call_stack)

        self.call_stack.pop()

    def visit_Block(self, node: Block):
        for declaration in node.declarations:
            self.visit(declaration)
        self.visit(node.compound_statement)

    def visit_ProcedureDecl(self, node: ProcedureDecl):
        pass

    def visit_FunctionDecl(self, node: FunctionDecl):
        pass

    def visit_VarDecl(self, node: VarDecl):
        var_name = node.var_node.value
        ar = self.call_stack.peek()
        # TODO: default value
        ar[var_name] = 0

    def visit_Param(self, node: Param):
        pass

    def visit_Type(self, node: Type):
        # TODO
        pass

    def visit_Compound(self, node: Compound):
        for stat in node.children:
            self.visit(stat)

    def visit_Call(self, node: Call):
        call_name   = node.name
        call_symbol = node.call_symbol

        if call_symbol is None:         # after semamic analysis, it must be a buildin function
            args = []
            for arg in node.arguments:
                args.append(self.visit(arg))
            ret_value = BUILDIN_FUNCTIONS[call_name](*args)
        elif type(call_symbol) in (ProcedureSymbol, FunctionSymbol):
            ar = ActivationRecord(call_name, ARType.PROCEDURE)

            for param, argument in zip(call_symbol.params, node.arguments):
                ar[param.name] = self.visit(argument)

            self.call_stack.push(ar)

            self.log(f'enter: CALL {call_name}')
            self.log(self.call_stack)

            self.visit(call_symbol.block_ast)

            self.log(f'leave: CALL {call_name}')
            self.log(self.call_stack)

            self.call_stack.pop()

            # 函数的返回值, 给函数同名的变量赋值, 此变量为返回值, 如无返回值, 该值为None
            ret_value = ar[call_name]
        else:
            self.error(node.token, ErrorInfo.id_not_callable(node.token.value))

        return ret_value

    def visit_Assign(self, node: Assign):
        var_name = node.left.value
        var_value = self.visit(node.right)

        ar = self.call_stack.peek()

        while ar is not None:
            if ar.has(var_name) or var_name == ar.name:     # for function return value
                ar[var_name] = var_value
                return
            else:
                ar = ar.enclosing

        self.error(node.token, ErrorInfo.id_not_valid(node.token))

    def visit_If(self, node: If):
        if self.visit(node.cond_expr):
            self.visit(node.then_stat)
        else:
            self.visit(node.else_stat)

    def visit_While(self, node: While):
        while self.visit(node.cond_expr):
            self.visit(node.do_stat)

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
        elif node.op.type == TokenType.EQUAL:
            return self.visit(node.left) == self.visit(node.right)
        elif node.op.type == TokenType.NOT_EQUAL:
            return self.visit(node.left) != self.visit(node.right)
        elif node.op.type == TokenType.GREATER:
            return self.visit(node.left) > self.visit(node.right)
        elif node.op.type == TokenType.LESS:
            return self.visit(node.left) < self.visit(node.right)
        else:
            self.error(node.op, ErrorInfo.unsupported_op(node.op.value))

    def visit_UnaryOp(self, node: UnaryOp):
        if node.op.type == TokenType.PLUS:
            return self.visit(node.expr)
        else:  # node.op.type == MINUS
            return -self.visit(node.expr)

    def visit_Var(self, node: Var):
        var_name = node.value
        ar = self.call_stack.peek()

        while ar is not None:
            var_value = ar.get(var_name)
            if var_value is not None:
                return var_value
            else:
                ar = ar.enclosing

        self.error(node.token, ErrorInfo.id_not_valid(node.token))

    def visit_String(self, node: String):
        return node.value

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
    global _SHOULD_LOG_STACK
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SPI - Simple Pascal Interpreter')
    # parser.add_argument('inputfile', help='Pascal source file')
    parser.add_argument('--scope', action='store_true', help='Print scope information')
    parser.add_argument('--stack', action='store_true', help='Print stack information')
    args = parser.parse_args()

    _SHOULD_LOG_SCOPE = args.scope
    _SHOULD_LOG_STACK = args.stack

    try:
        # spi_main()
        lexer = Lexer(test_code)
        parser = Parser(lexer)
        tree = parser.parse()

        displayer = Displayer(tree)
        displayer.display()

        semantic_analyzer = SemanticAnalyzer(tree)
        semantic_analyzer.analysis()

        interpreter = Interpreter(tree)
        interpreter.interpret()

    except (LexerError, ParserError, SemanticError, InterpreterError) as e:
        print(e)
        # raise e
