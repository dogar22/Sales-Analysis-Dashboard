import pandas as pd
import streamlit as st
import plotly.express as px
from queue import Queue
from io import BytesIO
import warnings
import time

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Node class for Linked List
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

# Linked List class for managing filtered data
class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node

    def to_list(self):
        result = []
        current = self.head
        while current:
            result.append(current.data)
            current = current.next
        return result

# Queue class for managing display order
class SimpleQueue:
    def __init__(self):
        self.queue = Queue()

    def enqueue(self, data):
        self.queue.put(data)

    def dequeue(self):
        return self.queue.get() if not self.queue.empty() else None

    def is_empty(self):
        return self.queue.empty()

# Load Dataset
def load_data(file_path):
    start_time = time.time()
    df = pd.read_excel(file_path, skiprows=4)
    print(f"load_data time complexity: O(n), Time Taken: {time.time() - start_time:.4f}s")
    return df

# Clean Dataset
def clean_data(df):
    start_time = time.time()
    required_columns = ['Invoice Date', 'Product', 'Total Sales', 'Operating Profit', 'Region']
    df = df[required_columns]
    df.columns = ['Date', 'Product', 'Sales', 'Profit', 'Region']
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Profit Margin (%)'] = (df['Profit'] / df['Sales']) * 100
    df['Profit Margin (%)'].fillna(0, inplace=True)
    print(f"clean_data time complexity: O(n), Time Taken: {time.time() - start_time:.4f}s")
    return df

# Generate Overview Tab
def generate_overview(df):
    start_time = time.time()
    st.subheader("Overview")
    total_sales = df['Sales'].sum()
    total_profit = df['Profit'].sum()
    avg_profit_margin = df['Profit Margin (%)'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales", f"${total_sales:,.2f}")
    col2.metric("Total Profit", f"${total_profit:,.2f}")
    col3.metric("Avg. Profit Margin", f"{avg_profit_margin:.2f}%")

    fig = px.bar(df, x='Date', y='Sales', title="Sales Over Time")
    st.plotly_chart(fig)
    print(f"generate_overview time complexity: O(n), Time Taken: {time.time() - start_time:.4f}s")

# Generate Trends Tab
def generate_trends(df):
    start_time = time.time()
    st.subheader("Sales Trends Over Time")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    date_range = df['Date'].dt.to_pydatetime()

    start_date, end_date = st.slider(
        "Select Date Range:",
        min_value=date_range.min(),
        max_value=date_range.max(),
        value=(date_range.min(), date_range.max())
    )

    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    fig = px.bar(filtered_df, x='Date', y='Sales', title="Sales Trends Over Time")
    st.plotly_chart(fig)
    print(f"generate_trends time complexity: O(n + k), Time Taken: {time.time() - start_time:.4f}s")

# Generate Product Performance Tab
def generate_product_performance(df):
    start_time = time.time()
    st.subheader("Product Performance")
    fig = px.scatter(df, x='Sales', y='Profit', color='Product', size='Profit Margin (%)',
                     title="Sales vs Profit by Product")
    st.plotly_chart(fig)
    print(f"generate_product_performance time complexity: O(n), Time Taken: {time.time() - start_time:.4f}s")

# Regional Performance with Hash Table
def generate_regional_performance(df):
    start_time = time.time()
    st.subheader("Regional Performance")

    regions_set = set(df['Region'].unique())
    region = st.selectbox("Select Region:", list(regions_set))

    filtered_data = df[df['Region'] == region]
    fig = px.pie(filtered_data, values='Sales', names='Product', title=f"Sales Distribution in {region}")
    st.plotly_chart(fig)
    print(f"generate_regional_performance time complexity: O(n + k), Time Taken: {time.time() - start_time:.4f}s")

# Search Products with Linked List and Queue
def search_product(df):
    start_time = time.time()
    st.subheader("Search Product by Name")

    product_name = st.text_input("Enter product name:")

    if product_name:
        linked_list = LinkedList()
        queue = SimpleQueue()

        filtered_df = df[df['Product'].str.contains(product_name, case=False, na=False)]

        for _, row in filtered_df.iterrows():
            linked_list.append(row.to_dict())
            queue.enqueue(row.to_dict())

        if linked_list.head:
            st.write(f"Filtered Results for '{product_name}':")
            organized_data = linked_list.to_list()
            organized_df = pd.DataFrame(organized_data)
            st.dataframe(organized_df)
        else:
            st.write("No matching products found.")
        print(f"search_product time complexity: O(n + k), Time Taken: {time.time() - start_time:.4f}s")

# Data Export
def export_data(df):
    start_time = time.time()
    output = BytesIO()
    df.to_excel(output, index=False)
    processed_data = output.getvalue()
    print(f"export_data time complexity: O(n), Time Taken: {time.time() - start_time:.4f}s")
    return processed_data

# Main Streamlit App
def main():
    st.title("Sales Data Analysis")

    uploaded_file = st.file_uploader("Upload your sales data file", type=["xlsx"])

    if uploaded_file:
        df = load_data(uploaded_file)
        df = clean_data(df)

        tabs = st.tabs(["Overview", "Trends", "Regional Performance", "Product Performance", "Search", "Export"])

        with tabs[0]:
            generate_overview(df)

        with tabs[1]:
            generate_trends(df)

        with tabs[2]:
            generate_regional_performance(df)

        with tabs[3]:
            generate_product_performance(df)

        with tabs[4]:
            search_product(df)

        with tabs[5]:
            st.subheader("Export Data")
            st.download_button(
                label="Download Cleaned Data",
                data=export_data(df),
                file_name="cleaned_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()
