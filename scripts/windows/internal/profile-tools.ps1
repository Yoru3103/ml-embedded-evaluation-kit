# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

function Get-MlekProfileCommand {
    param([hashtable]$Profiles, [string]$Name)

    if (-not $Profiles.ContainsKey($Name)) {
        throw "Unknown profile '$Name'. Available profiles: $($Profiles.Keys | Sort-Object)."
    }
    return Get-Command (Join-Path $PSScriptRoot "../$($Profiles[$Name])") -CommandType ExternalScript
}

function Get-MlekProfileParameters {
    param(
        [System.Management.Automation.ExternalScriptInfo]$Command,
        [string[]]$Exclude = @()
    )

    # Keep the parameter types and let the implementation choose its default parameter set.
    $parameters = [System.Management.Automation.RuntimeDefinedParameterDictionary]::new()
    foreach ($parameter in $Command.ScriptBlock.Ast.ParamBlock.Parameters) {
        $name = $parameter.Name.VariablePath.UserPath
        if ($name -in $Exclude) { continue }
        $definition = $Command.Parameters[$name]
        $attributes = [Collections.ObjectModel.Collection[Attribute]]::new()
        $attributes.Add([Management.Automation.ParameterAttribute]::new())
        foreach ($attribute in $definition.Attributes) {
            if ($attribute -isnot [Management.Automation.ParameterAttribute]) {
                $attributes.Add($attribute)
            }
        }
        $parameters.Add($name, [System.Management.Automation.RuntimeDefinedParameter]::new(
            $name, $definition.ParameterType, $attributes))
    }
    return $parameters
}
