import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

def load_features_from_json():
    path = filedialog.askopenfilename(title="Select advanced_features.json", filetypes=[("JSON Files", "*.json")])
    if not path:
        return None, None
    with open(path, "r") as f:
        data = json.load(f)
    return path, data

def save_features_to_json(json_path, data):
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    messagebox.showinfo("✅ Success", "Updated features saved!")

def show_editable_popup(json_path, features_data):
    win = tk.Tk()
    win.title("🎛️ Edit Audio Features")
    win.geometry("900x600")

    tree = ttk.Treeview(win, columns=("File", "Feature", "Value"), show="headings")
    tree.heading("File", text="File")
    tree.heading("Feature", text="Feature")
    tree.heading("Value", text="Value")
    tree.pack(fill="both", expand=True)

    for file, features in features_data.items():
        for k, v in features.items():
            tree.insert("", "end", values=(file, k, v))

    edit_frame = tk.Frame(win)
    edit_frame.pack(pady=10)

    tk.Label(edit_frame, text="Feature").grid(row=0, column=0)
    feature_entry = tk.Entry(edit_frame, width=30)
    feature_entry.grid(row=0, column=1)

    tk.Label(edit_frame, text="Value").grid(row=0, column=2)
    value_entry = tk.Entry(edit_frame, width=30)
    value_entry.grid(row=0, column=3)

    def on_row_select(event):
        selected = tree.focus()
        if not selected:
            return
        vals = tree.item(selected, "values")
        feature_entry.delete(0, tk.END)
        value_entry.delete(0, tk.END)
        feature_entry.insert(0, vals[1])
        value_entry.insert(0, vals[2])

    def update_feature():
        selected = tree.focus()
        if not selected:
            return
        vals = tree.item(selected, "values")
        new_feature = feature_entry.get()
        new_value = value_entry.get()
        tree.item(selected, values=(vals[0], new_feature, new_value))
        features_data[vals[0]].pop(vals[1], None)
        features_data[vals[0]][new_feature] = new_value

    def save_all():
        save_features_to_json(json_path, features_data)

    tree.bind("<<TreeviewSelect>>", on_row_select)
    tk.Button(edit_frame, text="Update", command=update_feature).grid(row=1, column=1, pady=10)
    tk.Button(edit_frame, text="Save All to JSON", command=save_all).grid(row=1, column=3, pady=10)

    win.mainloop()

if __name__ == "__main__":
    json_path, data = load_features_from_json()
    if json_path and data:
        show_editable_popup(json_path, data)
