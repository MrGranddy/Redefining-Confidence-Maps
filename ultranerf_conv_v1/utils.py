from typing import Annotated, Literal, TypeVar
import numpy as np
import numpy.typing as npt


DType = TypeVar("DType", bound=np.generic)
TrackingArray = Annotated[npt.NDArray[DType], Literal["N", 4, 4]]

def read_tracking(path: str) -> TrackingArray[np.float32]:
    """Reads an ImFusion tracking file.
        ImFusion tracking files are formatten in such a way that each line contains a number
        of values, only the first 16 of which are relevant. These 16 values are the entries
        of a 4x4 matrix, stored in column-major order.

        If ".ts" format, numbers are white-space separated.
        If ".csv" format, numbers are comma separated.
    
    Args:
        path: Path to the tracking file.

    Returns:
        Array of shape (N, 4, 4) containing the tracking data.
    """

    with open(path, 'r') as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines]

    # Remove empty lines.
    lines = [line for line in lines if line]

    # Parse lines.
    tracking = np.zeros((len(lines), 4, 4), dtype=np.float32)

    # Extension determines separator.
    if path.endswith('.ts'):
        lines = [line.split() for line in lines]
    elif path.endswith('.csv'):
        lines = [line.split(',') for line in lines]
    else:
        raise ValueError(f'Unknown file extension: {path}')

    for i, line in enumerate(lines):
        values = line[:16]

        for j in range(16):
            tracking[i, j % 4, j // 4] = float(values[j])

    return tracking

def write_tracking(path: str, tracking: TrackingArray[np.float32]) -> None:
    """Writes an ImFusion tracking file.
        ImFusion tracking files are formatten in such a way that each line contains a number
        of values, only the first 16 of which are relevant. These 16 values are the entries
        of a 4x4 matrix, stored in column-major order.

        If ".ts" format, numbers are white-space separated.
        If ".csv" format, numbers are comma separated.
    
    Args:
        path: Path to the tracking file.
        tracking: Array of shape (N, 4, 4) containing the tracking data.
    """

    if path.endswith('.ts'):
        seperator = ' '
    elif path.endswith('.csv'):
        seperator = ','
    else:
        raise ValueError(f'Unknown file extension: {path}')

    with open(path, 'w') as f:
        for i in range(tracking.shape[0]):

            values = [tracking[i, j % 4, j // 4] for j in range(16)]

            f.write(seperator.join([str(value) for value in values]))

            f.write('\n')

