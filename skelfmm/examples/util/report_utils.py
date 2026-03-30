def _format_tapply(report, i):
    unit = report.get("tapply_unit", "ms")
    value = report["tapply_seconds"][i]
    if unit == "ms":
        return f"{1e3 * value:.1f}", "Tapply_ms"
    return f"{value:.3e}", "Tapply_s"


def format_table(headers, rows):
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, item in enumerate(row):
            widths[i] = max(widths[i], len(str(item)))

    def fmt(row):
        return " | ".join(str(item).rjust(widths[i]) for i, item in enumerate(row))

    rule = "-+-".join("-" * width for width in widths)
    lines = [fmt(headers), rule]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def mesh_convergence_table(report):
    has_fmm = all(
        key in report for key in ("rank_max", "tbuild_seconds", "tapply_seconds", "storage_mb", "rel_subset_error")
    )
    has_corr = "tcorr_seconds" in report
    headers = ["N", "relerr"]
    if has_fmm:
        headers.extend(["fmm_relerr", "rank_max"])
        if has_corr:
            headers.append("Tcorr_s")
        _tapply_value, tapply_header = _format_tapply(report, 0)
        headers.extend(["Tbuild_s", tapply_header, "mem_mb"])
    headers.append("order")

    rows = []
    npanels = len(report["panels"])
    for i, panels in enumerate(report["panels"]):
        err = report["errors"][i] if i < npanels - 1 else "ref"
        order = f"{report['orders'][i]:.2f}" if i < len(report["orders"]) else "-"
        row = [panels, f"{err:.3e}" if isinstance(err, float) else err]
        if has_fmm:
            tapply_value, _ = _format_tapply(report, i)
            row.extend([f"{report['rel_subset_error'][i]:.3e}", report["rank_max"][i]])
            if has_corr:
                row.append(f"{report['tcorr_seconds'][i]:.3e}")
            row.extend(
                [
                    f"{report['tbuild_seconds'][i]:.3e}",
                    tapply_value,
                    f"{report['storage_mb'][i]:.1f}",
                ]
            )
        row.append(order)
        rows.append(tuple(row))

    return format_table(tuple(headers), rows)


def gmres_convergence_table(report):
    has_fmm = all(
        key in report for key in ("rank_max", "tbuild_seconds", "tapply_seconds", "storage_mb", "rel_subset_error")
    )
    has_corr = "tcorr_seconds" in report
    headers = ["N", "relerr", "gmres_iter"]
    if has_fmm:
        headers.extend(["fmm_relerr", "rank_max"])
        if has_corr:
            headers.append("Tcorr_s")
        _tapply_value, tapply_header = _format_tapply(report, 0)
        headers.extend(["Tbuild_s", tapply_header, "mem_mb"])
    headers.append("order")

    rows = []
    for i, panels in enumerate(report["panels"]):
        order = f"{report['orders'][i]:.2f}" if i < len(report["orders"]) else "-"
        row = [panels, f"{report['errors'][i]:.3e}", report["iterations"][i]]
        if has_fmm:
            tapply_value, _ = _format_tapply(report, i)
            row.extend([f"{report['rel_subset_error'][i]:.3e}", report["rank_max"][i]])
            if has_corr:
                row.append(f"{report['tcorr_seconds'][i]:.3e}")
            row.extend(
                [
                    f"{report['tbuild_seconds'][i]:.3e}",
                    tapply_value,
                    f"{report['storage_mb'][i]:.1f}",
                ]
            )
        row.append(order)
        rows.append(tuple(row))

    return format_table(tuple(headers), rows)
