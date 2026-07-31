# Core ML model assets

Generated Core ML packages are intentionally excluded from Git because the
current development set is approximately 864 MB and includes several
provisional models larger than GitHub's regular file limit.

Place locally converted `.mlpackage` directories in this folder before running
`xcodegen generate`. The workstation importer and Core ML conversion workflow
are documented in `WorkstationResults/`.

Once the final models have passed workstation and physical-device validation,
publish the selected packages as versioned release assets or store them with
Git LFS and document their checksums here.
