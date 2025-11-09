from ipywidgets import FileUpload, Output
from IPython.display import display
from exspy.signals import EELSSpectrum
import tempfile
import hyperspy.api as hs
import traceback

from .widgets import SpectraImageWidget


class EelsAnalyserWidget:
    """
    Interactive widget for uploading and analyzing EELS data files.

    This widget provides a file upload interface for EELS data files (dm4 or hspy formats).
    When a file is uploaded, it automatically loads the data using hyperspy and displays
    an interactive SpectraImageWidget for visualization and analysis.

    Attributes:
        _output: Output widget for displaying plots and error messages.
        _file_upload: File upload widget for selecting EELS data files.

    Example:
        >>> from mceels.widget import create_analyser_widget
        >>> create_analyser_widget()
    """
    def __init__(self):
        """
        Initialize the EelsAnalyserWidget.

        Creates the file upload widget and output display area. The file upload widget
        is configured to accept dm4 and hspy file formats and will trigger the analysis
        callback when a file is selected.
        """
        self._output: Output = Output()
        self._file_upload: FileUpload = FileUpload(
            accept="*.dm4, *.hspy",
            multiple=False,
            description="Upload EELS data"
        )
        self._file_upload.observe(self._on_file_upload, names='value')

    def create_widgets(self) -> None:
        """
        Display the file upload widget and output area in the notebook.

        This method should be called to render the widget in a Jupyter notebook.
        The file upload button will appear first, followed by the output area where
        analysis results will be displayed.
        """
        display(self._file_upload)
        display(self._output)

    def _on_file_upload(self, _) -> None:
        """
        Handle file upload events.

        This callback is triggered when a user selects a file. It loads the EELS data
        from the uploaded file and displays it using the SpectraImageWidget. Any errors
        during loading or display are caught and shown in the output area.

        Args:
            _: Event object (unused).
        """
        with self._output:
            self._output.clear_output(wait=True)

            try:
                spectra: EELSSpectrum = self._get_uploaded_file_contents()
                SpectraImageWidget(spectra).show()
            except Exception as e:
                print(f"Error: {str(e)}")
                traceback.print_exc()

    def _get_uploaded_file_contents(self) -> EELSSpectrum:
        """
        Load EELS data from the uploaded file.

        This method retrieves the uploaded file data, writes it to a temporary file,
        and loads it using hyperspy. The temporary file is automatically cleaned up
        after loading.

        Returns:
            EELSSpectrum object containing the loaded EELS data.

        Raises:
            RuntimeError: If no file has been uploaded.
        """
        if not self._file_upload.value:
            raise RuntimeError("No file detected!")

        uploaded_file: dict = self._file_upload.value[0]
        file_name: str = uploaded_file['name']

        file_contents: bytes = uploaded_file['content']
        with tempfile.NamedTemporaryFile(suffix=f"-{file_name}") as file:
            file.write(file_contents)
            loaded_data = hs.load(file.name)

        if isinstance(loaded_data, EELSSpectrum):
            return loaded_data
        if isinstance(loaded_data, (list, tuple)):
            return loaded_data[3]
        

def create_analyser_widget() -> None:
    """
    Create and display an EELS analyzer widget in a Jupyter notebook.

    This is a convenience function that creates an EelsAnalyserWidget instance
    and immediately displays it. Users can upload dm4 or hspy EELS data files
    which will be automatically loaded and visualized.

    Example:
        >>> import mceels
        >>> mceels.create_analyser_widget()
        # A file upload button will appear. After uploading a file,
        # an interactive plot with spatial heatmap and spectrum will be displayed.
    """
    widget = EelsAnalyserWidget()
    widget.create_widgets()