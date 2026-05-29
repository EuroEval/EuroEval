# Tool calling

## 📚 Overview

Tool calling is a task of using the right functions (picked from a list) and
using them with the right arguments - understanding the user request.

We use a uniform custom prompt and structured JSON output generation to obtain
predictions from a given model.

When evaluating generative models, we allow the model to generate 500 tokens on this
task.

## 📊 Metrics

We use simple accuracy where a prediction is positive only if the list of functions
and required arguments exactly matches one of the options given in the ground truth.
For example, given the ground truth:

```json
[
    {
        "latest_exchange_rate": {
            "source_currency": [
                "USD",
                "US Dollars",
                "US Dollar"
            ],
            "target_currency": [
                "EUR",
                "Euro"
            ],
            "amount": [
                1000
            ]
        }
    },
    {
        "safeway.order": {
            "location": [
                "Palo Alto, CA",
                "Palo Alto",
                "CA"
            ],
            "items": [
                [
                    "water",
                    "apples",
                    "bread"
                ]
            ],
            "quantity": [
                [
                    2,
                    3,
                    1
                ]
            ]
        }
    },
]
```

and function descriptions:

```json
[
    {
        "name": "safeway.order",
        "description": "Order specified items from a Safeway location.",
        "parameters": {
            "type": "dict",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location of the Safeway store, e.g. Palo Alto, CA."
                },
                "items": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of items to order."
                },
                "quantity": {
                    "type": "array",
                    "items": {
                        "type": "integer"
                    },
                    "description": "Quantity of each item in the order list."
                }
            },
            "required": [
                "location",
                "items",
                "quantity"
            ]
        }
    },
    {
        "name": "latest_exchange_rate",
        "description": "Retrieve the latest exchange rate between two specified currencies.",
        "parameters": {
            "type": "dict",
            "properties": {
                "source_currency": {
                    "type": "string",
                    "description": "The currency you are converting from."
                },
                "target_currency": {
                    "type": "string",
                    "description": "The currency you are converting to."
                },
                "amount": {
                    "type": "integer",
                    "description": "The amount to be converted. If omitted, default to xchange rate of 1 unit source currency."
                }
            },
            "required": [
                "source_currency",
                "target_currency"
            ]
        }
    }
]
```

One valid prediction option is

```json
{
    "tool_calls": [
        {
            "function": "last_exchange_rate",
            "arguments": {
                "source_currency": "US Dollar", # or 'USD' or 'US Dollars'
                "target_currency": "EUR", # or 'Euro'
                "amount": 1000 # this can even be left out
            }
        },
        {
            "function": "safeway.order",
            "arguments": {
                "location": "Palo Alto",
                "items": ["water", "apples", "bread"],
                "quantity": [2, 3, 1]
            }
        }
    ]
}
```

## 🛠️ How to run

In the command line interface of the [EuroEval Python package](/python-package), you
can benchmark your favorite model on the tool calling task like so:

```bash
euroeval --model <model-id> --task tool-calling
```
