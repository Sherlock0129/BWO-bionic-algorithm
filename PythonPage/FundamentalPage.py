import streamlit as st

def fundamental_page():
    # Page title
    st.title("Python Basics")

    # Variables and Data Types
    st.header("Variables and Data Types")
    st.markdown("""
    In Python, variables don't need type declarations and are defined by assignment.
    Common data types include:
    - Integer (`int`)
    - Floating-point (`float`)
    - String (`str`)
    - Boolean (`bool`)
    """)

    st.code("""
    # Example
    a = 10          # Integer
    b = 3.14        # Float
    c = "Hello"     # String
    d = True        # Boolean
    """, language="python")

    # Lists
    st.header("Lists")
    st.markdown("""
    Lists are ordered collections in Python that can contain multiple data types and allow element addition, deletion, etc.
    """)

    st.code("""
    # Example
    my_list = [1, 2, 3, 4, 5]
    my_list.append(6)
    print(my_list)  # Output: [1, 2, 3, 4, 5, 6]
    """, language="python")

    # Sets
    st.header("Sets")
    st.markdown("""
    Sets are unordered collections of unique elements, ideal for removing duplicates.
    """)

    st.code("""
    # Example
    my_set = {1, 2, 3, 3, 4}
    print(my_set)  # Output: {1, 2, 3, 4}
    """, language="python")

    # Dictionaries
    st.header("Dictionaries")
    st.markdown("""
    Dictionaries (`dict`) are key-value pairs and allow quick lookups by keys.
    """)

    st.code("""
    # Example
    my_dict = {"name": "Alice", "age": 25}
    print(my_dict["name"])  # Output: "Alice"
    """, language="python")

    # Basic Operators
    st.header("Basic Operators")
    st.markdown("""
    Python supports various operators:
    - Arithmetic operators: `+`, `-`, `*`, `/`, `%`, `**`, `//`
    - Comparison operators: `==`, `!=`, `>`, `<`, `>=`, `<=`
    - Logical operators: `and`, `or`, `not`
    """)

    st.code("""
    # Example
    x = 10
    y = 3
    print(x + y)    # Addition, Output: 13
    print(x > y)    # Comparison, Output: True
    """, language="python")

    # Loops
    st.header("Loops")
    st.markdown("""
    Python commonly uses `for` and `while` loops to iterate over sequences or repeat actions.
    """)

    st.code("""
    # Example
    # for loop
    for i in range(5):
        print(i)    # Output: 0, 1, 2, 3, 4
    
    # while loop
    count = 0
    while count < 5:
        print(count)
        count += 1
    """, language="python")

    # Control Flow
    st.header("Control Flow")
    st.markdown("""
    Control flow statements include `if`, `elif`, and `else` for conditional execution.
    """)

    st.code("""
    # Example
    x = 10
    if x > 5:
        print("x is greater than 5")
    elif x == 5:
        print("x is equal to 5")
    else:
        print("x is less than 5")
    """, language="python")
