#ifndef TEMPLATE_COMMON_H
#define TEMPLATE_COMMON_H

#include "../core/core.h"

# define EMPTY(...)
# define DEFER(...) __VA_ARGS__ EMPTY()
# define OBSTRUCT(...) __VA_ARGS__ DEFER(EMPTY)()
# define EXPAND(...) __VA_ARGS__

#define FOR_BOOL(expr, ...) \
expr(cbool, __VA_ARGS__)

#define FORALL_INT_DTYPES(expr, ...) \
expr(cbool, __VA_ARGS__) \
expr(int8, __VA_ARGS__) \
expr(uint8, __VA_ARGS__) \
expr(int16, __VA_ARGS__) \
expr(uint16, __VA_ARGS__) \
expr(int32, __VA_ARGS__) \
expr(uint32, __VA_ARGS__) \
expr(int64, __VA_ARGS__) \
expr(uint64, __VA_ARGS__)

#define FORALL_FLOAT_DTYPES(expr, ...) \
expr(float32, __VA_ARGS__) \
expr(float64, __VA_ARGS__)

#define FORALL_DTYPES(expr, ...) \
FORALL_INT_DTYPES(expr, __VA_ARGS__) \
FORALL_FLOAT_DTYPES(expr, __VA_ARGS__)

#define FUNC_WRAPPER(dtype_iterator, macro, name) \
macro##_PROTO(name) \
{ \
    switch(dtype) { \
        dtype_iterator(macro##_CASE, name) \
        default: \
            exit(1); \
    } \
}

#define FUNC(macro, name, ...) \
FORALL_DTYPES(macro, name, __VA_ARGS__) \
\
FUNC_WRAPPER(FORALL_DTYPES, macro, name)

#define FUNC_GENERIC(macro, name, ...) \
FORALL_DTYPES(macro##_GENERIC, name, __VA_ARGS__) \
\
FUNC_WRAPPER(FORALL_DTYPES, macro, name)

#define FUNC_INT_FLOAT(macro, name, function) \
FORALL_INT_DTYPES(macro, name, function##_INT) \
FORALL_FLOAT_DTYPES(macro, name, function##_FLOAT) \
\
FUNC_WRAPPER(FORALL_DTYPES, macro, name)

#define FUNC_INT_FLOAT32_64(macro, name, function) \
FORALL_INT_DTYPES(macro, name, function##_INT) \
macro(float32, name, function##_FLOAT32) \
macro(float64, name, function##_FLOAT64) \
\
FUNC_WRAPPER(FORALL_DTYPES, macro, name)

#define FUNC_MATH(macro, name) \
macro(float32, name, name##f) \
macro(float64, name, name) \
\
FUNC_WRAPPER(FORALL_FLOAT_DTYPES, macro, name)

#endif
