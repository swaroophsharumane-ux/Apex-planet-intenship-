import pandas as pd
import numpy as np
import os
import re

# ============================================================
# APEXPLANET DATA ANALYTICS INTERNSHIP
# TASK 1: DATA ACQUISITION, CLEANING & PREPARATION
# ============================================================

# ============================================================
# 1. FILE LOCATIONS
# ============================================================

INPUT_FILE = r"C:\PythonProject3\ApexPlanet_DataAnalytics_Dataset.xlsx"

OUTPUT_FOLDER = r"C:\PythonProject3\ApexPlanet_Task1_Output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

CLEANED_FILE = os.path.join(
    OUTPUT_FOLDER,
    "ApexPlanet_Cleaned_Dataset.xlsx"
)

DICTIONARY_FILE = os.path.join(
    OUTPUT_FOLDER,
    "ApexPlanet_Data_Dictionary.xlsx"
)

PROFILING_FILE = os.path.join(
    OUTPUT_FOLDER,
    "ApexPlanet_Profiling_Report.xlsx"
)

OUTLIER_FILE = os.path.join(
    OUTPUT_FOLDER,
    "ApexPlanet_Outlier_Report.xlsx"
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("APEXPLANET DATA ANALYTICS - TASK 1")
print("=" * 70)

print("\nLoading Excel dataset...")

try:
    df = pd.read_excel(INPUT_FILE)

except FileNotFoundError:
    print("\nERROR: Excel file not found.")
    print("Check this path:")
    print(INPUT_FILE)
    exit()

except Exception as e:
    print("\nERROR while loading Excel file:")
    print(e)
    exit()


print("\nDATASET LOADED SUCCESSFULLY")

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ============================================================
# 3. KEEP ORIGINAL COPY
# ============================================================

original_df = df.copy()

original_rows = len(df)
original_columns = len(df.columns)


# ============================================================
# 4. DISPLAY FIRST RECORDS
# ============================================================

print("\n" + "=" * 70)
print("FIRST 5 RECORDS")
print("=" * 70)

print(df.head())


# ============================================================
# 5. STANDARDIZE COLUMN NAMES
# ============================================================

print("\n" + "=" * 70)
print("STANDARDIZING COLUMN NAMES")
print("=" * 70)

old_columns = df.columns.tolist()


def clean_column_name(column):

    column = str(column)

    column = column.strip()

    column = column.lower()

    column = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        column
    )

    column = column.strip("_")

    return column


df.columns = [
    clean_column_name(column)
    for column in df.columns
]


print("Original columns:")
print(old_columns)

print("\nStandardized columns:")
print(df.columns.tolist())


# ============================================================
# 6. BASIC DATA PROFILING
# ============================================================

print("\n" + "=" * 70)
print("BASIC DATA PROFILING")
print("=" * 70)

print("Number of rows    :", len(df))
print("Number of columns :", len(df.columns))
print("Total cells       :", df.size)

print("\nData types:")

print(df.dtypes)


# ============================================================
# 7. MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE ANALYSIS")
print("=" * 70)

missing_count = df.isnull().sum()

missing_percentage = (
    df.isnull().sum()
    / len(df)
    * 100
).round(2)


missing_report = pd.DataFrame({

    "Column":
        df.columns,

    "Missing_Values":
        missing_count.values,

    "Missing_Percentage":
        missing_percentage.values
})


print(
    missing_report.to_string(
        index=False
    )
)


# ============================================================
# 8. DUPLICATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE ANALYSIS")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print(
    "Duplicate rows found:",
    duplicate_count
)


# ============================================================
# 9. UNIQUE VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("UNIQUE VALUE ANALYSIS")
print("=" * 70)

for column in df.columns:

    print(
        column,
        "->",
        df[column].nunique(
            dropna=True
        ),
        "unique values"
    )


# ============================================================
# 10. CATEGORICAL DATA PROFILING
# ============================================================

print("\n" + "=" * 70)
print("CATEGORICAL DATA PROFILING")
print("=" * 70)

categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns


for column in categorical_columns:

    print("\nColumn:", column)

    print(
        df[column]
        .value_counts(
            dropna=False
        )
        .head(10)
    )


# ============================================================
# 11. NUMERICAL DATA PROFILING
# ============================================================

print("\n" + "=" * 70)
print("NUMERICAL DATA PROFILING")
print("=" * 70)

numeric_columns = df.select_dtypes(
    include=np.number
).columns


if len(numeric_columns) > 0:

    print(
        df[numeric_columns]
        .describe()
        .transpose()
    )

else:

    print(
        "No numerical columns found."
    )


# ============================================================
# 12. AUTOMATIC DATE DETECTION
# ============================================================

print("\n" + "=" * 70)
print("DATE DETECTION AND STANDARDIZATION")
print("=" * 70)

date_columns = []


for column in df.columns:

    column_name = column.lower()

    if any(
        word in column_name
        for word in [
            "date",
            "dob",
            "birth",
            "time"
        ]
    ):

        converted = pd.to_datetime(
            df[column],
            errors="coerce"
        )

        valid_dates = converted.notna().sum()

        if valid_dates > 0:

            df[column] = converted

            date_columns.append(column)

            print(
                column,
                "-> converted to datetime"
            )


if len(date_columns) == 0:

    print(
        "No date columns automatically detected."
    )


# ============================================================
# 13. REMOVE DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE CLEANING")
print("=" * 70)

rows_before = len(df)

df = df.drop_duplicates()

rows_after = len(df)

duplicates_removed = (
    rows_before - rows_after
)


print(
    "Rows before:",
    rows_before
)

print(
    "Rows after:",
    rows_after
)

print(
    "Duplicates removed:",
    duplicates_removed
)


# ============================================================
# 14. HANDLE MISSING NUMERICAL VALUES
# ============================================================

print("\n" + "=" * 70)
print("NUMERICAL MISSING VALUE CLEANING")
print("=" * 70)

numeric_columns = df.select_dtypes(
    include=np.number
).columns


for column in numeric_columns:

    missing = df[column].isnull().sum()

    if missing > 0:

        median_value = df[column].median()

        df[column] = df[column].fillna(
            median_value
        )

        print(
            column,
            "->",
            missing,
            "missing values filled with median:",
            median_value
        )


# ============================================================
# 15. HANDLE MISSING CATEGORICAL VALUES
# ============================================================

print("\n" + "=" * 70)
print("CATEGORICAL MISSING VALUE CLEANING")
print("=" * 70)

categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns


for column in categorical_columns:

    missing = df[column].isnull().sum()

    if missing > 0:

        mode_values = df[column].mode()

        if len(mode_values) > 0:

            mode_value = mode_values.iloc[0]

            df[column] = df[column].fillna(
                mode_value
            )

            print(
                column,
                "->",
                missing,
                "missing values filled with:",
                mode_value
            )

        else:

            df[column] = df[column].fillna(
                "Unknown"
            )


# ============================================================
# 16. CLEAN TEXT FORMATTING
# ============================================================

print("\n" + "=" * 70)
print("TEXT FORMATTING CLEANING")
print("=" * 70)

text_columns = df.select_dtypes(
    include=["object"]
).columns


for column in text_columns:

    df[column] = df[column].apply(
        lambda x:
        x.strip()
        if isinstance(x, str)
        else x
    )


print(
    "Leading and trailing spaces removed."
)


# ============================================================
# 17. STANDARDIZE CATEGORICAL TEXT
# ============================================================

print("\n" + "=" * 70)
print("CATEGORICAL STANDARDIZATION")
print("=" * 70)


for column in text_columns:

    unique_count = df[column].nunique()

    # Only apply title case to
    # relatively small categorical fields

    if unique_count <= 50:

        df[column] = df[column].apply(
            lambda x:
            x.title()
            if isinstance(x, str)
            else x
        )

        print(
            column,
            "-> standardized"
        )


# ============================================================
# 18. FEATURE ENGINEERING - CUSTOMER AGE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

dob_column = None


for column in df.columns:

    column_name = column.lower()

    if (
        column_name == "dob"
        or
        "date_of_birth" in column_name
        or
        "birth_date" in column_name
        or
        "dateofbirth" in column_name
    ):

        dob_column = column

        break


if dob_column is not None:

    today = pd.Timestamp.today()

    df["customer_age"] = (
        today.year
        -
        df[dob_column].dt.year
        -
        (
            (
                today.month,
                today.day
            )
            <
            (
                df[dob_column].dt.month,
                df[dob_column].dt.day
            )
        )
    )

    print(
        "Customer Age created from:",
        dob_column
    )

else:

    print(
        "Date of Birth column not found."
    )


# ============================================================
# 19. AGE GROUP
# ============================================================

if "customer_age" in df.columns:

    df["age_group"] = pd.cut(

        df["customer_age"],

        bins=[
            0,
            18,
            25,
            35,
            50,
            65,
            120
        ],

        labels=[
            "Below 18",
            "18-25",
            "26-35",
            "36-50",
            "51-65",
            "65+"
        ],

        include_lowest=True
    )

    print(
        "Age Group created."
    )


# ============================================================
# 20. ORDER MONTH AND YEAR
# ============================================================

for column in date_columns:

    if "order" in column.lower():

        df["order_month"] = (
            df[column]
            .dt.month_name()
        )

        df["order_year"] = (
            df[column]
            .dt.year
        )

        print(
            "Order Month created."
        )

        print(
            "Order Year created."
        )

        break


# ============================================================
# 21. OUTLIER DETECTION - IQR METHOD
# ============================================================

print("\n" + "=" * 70)
print("OUTLIER DETECTION")
print("=" * 70)

numeric_columns = df.select_dtypes(
    include=np.number
).columns


outlier_results = []


for column in numeric_columns:

    Q1 = df[column].quantile(
        0.25
    )

    Q3 = df[column].quantile(
        0.75
    )

    IQR = Q3 - Q1

    lower_limit = (
        Q1 - 1.5 * IQR
    )

    upper_limit = (
        Q3 + 1.5 * IQR
    )

    outlier_mask = (
        (df[column] < lower_limit)
        |
        (df[column] > upper_limit)
    )

    outlier_count = (
        outlier_mask.sum()
    )

    outlier_percentage = (
        outlier_count
        /
        len(df)
        *
        100
    )

    outlier_results.append({

        "Column":
            column,

        "Q1":
            Q1,

        "Q3":
            Q3,

        "IQR":
            IQR,

        "Lower_Limit":
            lower_limit,

        "Upper_Limit":
            upper_limit,

        "Outlier_Count":
            outlier_count,

        "Outlier_Percentage":
            round(
                outlier_percentage,
                2
            )
    })

    print(
        column,
        "->",
        outlier_count,
        "potential outliers"
    )


outlier_report = pd.DataFrame(
    outlier_results
)


# ============================================================
# 22. FLAG OUTLIERS
# ============================================================

print("\nCreating outlier flags...")


for column in numeric_columns:

    Q1 = df[column].quantile(
        0.25
    )

    Q3 = df[column].quantile(
        0.75
    )

    IQR = Q3 - Q1

    lower_limit = (
        Q1 - 1.5 * IQR
    )

    upper_limit = (
        Q3 + 1.5 * IQR
    )

    df[column + "_outlier"] = np.where(

        (
            (df[column] < lower_limit)
            |
            (df[column] > upper_limit)
        ),

        "Yes",

        "No"
    )


print(
    "Outlier flags created."
)


# ============================================================
# 23. FINAL DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATA QUALITY CHECK")
print("=" * 70)

final_missing = df.isnull().sum()

total_final_missing = (
    final_missing.sum()
)

final_duplicates = (
    df.duplicated().sum()
)


print(
    "Final Rows:",
    len(df)
)

print(
    "Final Columns:",
    len(df.columns)
)

print(
    "Remaining Missing Values:",
    total_final_missing
)

print(
    "Remaining Duplicate Rows:",
    final_duplicates
)


# ============================================================
# 24. CREATE DATA DICTIONARY
# ============================================================

print("\n" + "=" * 70)
print("CREATING DATA DICTIONARY")
print("=" * 70)


dictionary_data = []


for column in df.columns:

    pandas_type = str(
        df[column].dtype
    )

    unique_values = (
        df[column]
        .nunique(
            dropna=True
        )
    )

    missing_values = (
        df[column].isnull().sum()
    )


    if pd.api.types.is_numeric_dtype(
        df[column]
    ):

        data_type = "Numerical"

    elif pd.api.types.is_datetime64_any_dtype(
        df[column]
    ):

        data_type = "Date/Time"

    else:

        data_type = "Categorical/Text"


    # Generate a readable variable meaning

    variable_meaning = (
        column
        .replace("_", " ")
        .title()
    )


    # Generate business relevance

    business_relevance = (
        "Can be used for analysis, "
        "reporting and business insights."
    )


    dictionary_data.append({

        "Variable_Name":
            column,

        "Meaning":
            variable_meaning,

        "Data_Type":
            data_type,

        "Pandas_Data_Type":
            pandas_type,

        "Unique_Values":
            unique_values,

        "Missing_Values":
            missing_values,

        "Business_Relevance":
            business_relevance

    })


data_dictionary = pd.DataFrame(
    dictionary_data
)


# ============================================================
# 25. CREATE PROFILING REPORT
# ============================================================

print("\nCreating profiling report...")


profiling_summary = pd.DataFrame({

    "Metric": [

        "Original Rows",

        "Original Columns",

        "Final Rows",

        "Final Columns",

        "Duplicate Rows Found",

        "Duplicate Rows Removed",

        "Final Missing Values",

        "Final Duplicate Rows"

    ],

    "Value": [

        original_rows,

        original_columns,

        len(df),

        len(df.columns),

        duplicate_count,

        duplicates_removed,

        total_final_missing,

        final_duplicates

    ]

})


# ============================================================
# 26. SAVE CLEANED DATASET
# ============================================================

print("\n" + "=" * 70)
print("SAVING FINAL DATASET")
print("=" * 70)


df.to_excel(
    CLEANED_FILE,
    index=False
)


print(
    "Cleaned dataset saved:"
)

print(
    CLEANED_FILE
)


# ============================================================
# 27. SAVE DATA DICTIONARY
# ============================================================

data_dictionary.to_excel(

    DICTIONARY_FILE,

    index=False
)


print(
    "\nData dictionary saved:"
)

print(
    DICTIONARY_FILE
)


# ============================================================
# 28. SAVE PROFILING REPORT
# ============================================================

profiling_summary.to_excel(

    PROFILING_FILE,

    index=False
)


print(
    "\nProfiling report saved:"
)

print(
    PROFILING_FILE
)


# ============================================================
# 29. SAVE OUTLIER REPORT
# ============================================================

outlier_report.to_excel(

    OUTLIER_FILE,

    index=False
)


print(
    "\nOutlier report saved:"
)

print(
    OUTLIER_FILE
)


# ============================================================
# 30. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TASK 1 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nINPUT:")
print(INPUT_FILE)

print("\nOUTPUT FOLDER:")
print(OUTPUT_FOLDER)

print("\nFILES CREATED:")

print(
    "1. ApexPlanet_Cleaned_Dataset.xlsx"
)

print(
    "2. ApexPlanet_Data_Dictionary.xlsx"
)

print(
    "3. ApexPlanet_Profiling_Report.xlsx"
)

print(
    "4. ApexPlanet_Outlier_Report.xlsx"
)

print("\nFINAL DATASET INFORMATION:")

print(
    "Rows    :",
    len(df)
)

print(
    "Columns :",
    len(df.columns)
)

print(
    "Missing :",
    total_final_missing
)

print(
    "Duplicates :",
    final_duplicates
)

print(
    "\nDataset is ready for analysis."
)

print("=" * 70)
