import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
from qc_core.paths import BASE_DIR

from qc_core.pipeline import analyze_mxf
from qc_core.paths import ensure_dirs

LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo_ta.png")


class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("ANALIZADOR DE AUDIO CON IA - TELEAMAZONAS")
        self.geometry("680x620")

        ensure_dirs()

        self.mxf_path = tk.StringVar(value="")
        self.selected_files = []
        self.model_key = tk.StringVar(value="logreg")
        self.output_dir = tk.StringVar(value="")

        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        # Encabezado visual
        header = ttk.Frame(frm)
        header.grid(row=0, column=0, columnspan=4, pady=(10, 20), sticky="n")

        # Título principal
        title_lbl = ttk.Label(
            header,
            text="ANALIZADOR DE AUDIO CON IA",
            font=("Arial", 20, "bold")
        )
        title_lbl.pack(pady=(0, 10))

        # Logo
        self.logo_img = None
        if os.path.exists(LOGO_PATH):
            try:
                img = Image.open(LOGO_PATH)
                img = img.resize((100, 70))  # ajustable
                self.logo_img = ImageTk.PhotoImage(img)
                logo_lbl = ttk.Label(header, image=self.logo_img)
                logo_lbl.pack(pady=(0, 10))
            except Exception:
                pass

        # Subtítulo
        subtitle_lbl = ttk.Label(
            header,
            text="Sistema automatizado de control de calidad de audio broadcast\nbasado en normativa EBU R128 y modelos de inteligencia artificial",
            font=("Arial", 10),
            justify="center"
        )
        subtitle_lbl.pack()

        #ttk.Label(frm, text="Archivo MXF/WAV:").grid(row=1, column=0, sticky="w")
        #ent = ttk.Entry(frm, textvariable=self.mxf_path, width=90)
        #ent.grid(row=2, column=0, columnspan=3, sticky="we", pady=(4, 8))

        #Boton para seleccionar archivos MXF
        ttk.Button(frm, text="Seleccionar archivos MXF", command=self.pick_mxf).grid(row=1, column=0, columnspan=4, pady=(12, 0))
      
        # Lista visual de archivos seleccionados
        files_frame = ttk.Frame(frm)
        files_frame.grid(row=3, column=0, columnspan=4, sticky="nsew", pady=(4, 8))

        self.files_listbox = tk.Listbox(files_frame, height=6, width=110)
        self.files_listbox.pack(side="left", fill="both", expand=True)

        files_scroll = ttk.Scrollbar(files_frame, orient="vertical", command=self.files_listbox.yview)
        files_scroll.pack(side="right", fill="y")

        self.files_listbox.config(yscrollcommand=files_scroll.set)

        #Boton para limpiar lista
        ttk.Button(frm, text="Limpiar lista", command=self.clear_file_list).grid(row=4, column=0, columnspan=4, pady=(12, 0))
        
        # Fila propia para el selector de modelo
        model_row = ttk.Frame(frm)
        model_row.grid(row=5, column=0, columnspan=4, sticky="w", pady=(4, 0))

        ttk.Label(model_row, text="Modelo IA:").pack(side="left")

        cmb = ttk.Combobox(
            model_row,
            textvariable=self.model_key,
            values=[
                "logreg - Regresión Logística (baseline)",
                "rf - Random Forest (comparativo)",
            ],
            state="readonly",
            width=34
        )
        cmb.pack(side="left", padx=(8, 0))
        cmb.current(0)

        #Fila visual para ruta de reportes
        ttk.Label(frm, text="Carpeta de reportes:").grid(row=6, column=0, sticky="w")

        self.output_entry = ttk.Entry(frm, textvariable=self.output_dir, width=90)
        self.output_entry.grid(row=7, column=0, columnspan=3, sticky="we", pady=(4, 8))

        ttk.Button(frm, text="Seleccionar carpeta", command=self.pick_output_dir).grid(row=8, column=0, columnspan=4, pady=(12, 0))



        #Boton Analizar y generar reporte
        self.btn_run = ttk.Button(frm, text="Analizar y generar reportes", command=self.run_analysis)
        self.btn_run.grid(row=9, column=0, columnspan=4, pady=(12, 0))

        self.progress = ttk.Progressbar(frm, mode="indeterminate")
        self.progress.grid(row=10, column=0, columnspan=4, sticky="we", pady=(12, 6))

        self.status = tk.StringVar(value="Listo.")
        ttk.Label(frm, textvariable=self.status).grid(row=11, column=0, columnspan=4, sticky="w")

        frm.columnconfigure(0, weight=1)
        frm.columnconfigure(1, weight=0)
        frm.columnconfigure(2, weight=0)
        frm.columnconfigure(3, weight=1)

        self.check_requirements()

    def check_requirements(self):
        from qc_core.paths import FFMPEG_EXE, FFPROBE_EXE, MODEL_FILES
        missing = []
        if not os.path.exists(FFMPEG_EXE):
            missing.append("ffmpeg/ffmpeg.exe")
        if not os.path.exists(FFPROBE_EXE):
            missing.append("ffmpeg/ffprobe.exe")

        # Validar ambos modelos
        for key, path in MODEL_FILES.items():
            if not os.path.exists(path):
                missing.append(f"model/{os.path.basename(path)}")

        if missing:
            messagebox.showwarning(
                "Faltan componentes",
                "Faltan componentes requeridos:\n- " + "\n- ".join(missing) +
                "\n\nLa app no podrá analizar hasta que estén presentes."
            )

    def pick_mxf(self):
        paths = filedialog.askopenfilenames(
            title="Seleccionar archivos MXF o WAV",
            filetypes=[
                ("Audio/Video", "*.mxf *.wav"),
                ("MXF files", "*.mxf"),
                ("WAV files", "*.wav"),
                ("All files", "*.*")
            ]
        )

        if paths:
            self.selected_files = list(paths)

            # Actualizar campo resumen
            if len(self.selected_files) == 1:
                self.mxf_path.set(self.selected_files[0])
            else:
                self.mxf_path.set(f"{len(self.selected_files)} archivos seleccionados")

            # Actualizar lista visual
            self.files_listbox.delete(0, tk.END)
            for path in self.selected_files:
                self.files_listbox.insert(tk.END, os.path.basename(path))

    def run_analysis(self):
        if not self.selected_files:
            # compatibilidad por si alguien escribió manualmente una ruta
            single = self.mxf_path.get().strip()
            if single and os.path.exists(single):
                self.selected_files = [single]

        if not self.selected_files:
            messagebox.showerror("Error", "Selecciona al menos un archivo MXF o WAV válido.")
            return

        self.btn_run.config(state="disabled")
        self.progress.start(10)
        self.status.set(f"Analizando {len(self.selected_files)} archivo(s)…")

        self.after(100, self._do_analysis)

    def _do_analysis(self):
        try:
            sel = self.model_key.get().split(" - ")[0].strip()  # "logreg" o "rf"

            processed = 0
            failed = 0
            ok_files = 0
            review_files = 0
            generated_reports = []

            total = len(self.selected_files)

            for i, path in enumerate(self.selected_files, start=1):
                self.status.set(f"Analizando archivo {i}/{total}: {os.path.basename(path)}")
                self.update_idletasks()

                try:
                    report_path, decision, ebu = analyze_mxf(
                        path,
                        segment_s=5.0,
                        model_key=sel,
                        output_dir=self.output_dir.get().strip() or None
                    )
                    generated_reports.append(report_path)
                    processed += 1

                    if decision["final_requires_review"]:
                        review_files += 1
                    else:
                        ok_files += 1

                except Exception:
                    failed += 1

            summary = (
                f"Proceso finalizado.\n\n"
                f"Archivos procesados: {processed}\n"
                f"OK: {ok_files}\n"
                f"Requieren revisión: {review_files}\n"
                f"Errores: {failed}"
            )

            messagebox.showinfo("Listo", summary)
            self.status.set("Proceso finalizado correctamente.")

            # Abrir el último reporte generado
            if generated_reports:
                try:
                    os.startfile(generated_reports[-1])
                except Exception:
                    pass

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status.set(f"Error: {e}")

        finally:
            self.progress.stop()
            self.btn_run.config(state="normal")

    #Funcion para lista de archivos seleccionados
    def clear_file_list(self):
        self.selected_files = []
        self.mxf_path.set("")
        self.files_listbox.delete(0, tk.END)

    #Funcion para elegir carpeta de ruta de los reportes generados
    def pick_output_dir(self):
        folder = filedialog.askdirectory(title="Seleccionar carpeta para guardar reportes")
        if folder:
            self.output_dir.set(folder)


if __name__ == "__main__":
    App().mainloop()