var RBM = {};
RBM.Create = function(inDimensionsIn, inDimensionsOut)
{
    var obj = {};
    var min = [];
    var max = [];
    var i;
    
    inDimensionsIn++;
    inDimensionsOut++;

    for(i=0; i<inDimensionsIn; i++)
    {
        min.push(-0.1);
        max.push(0.1);
    }
    
    obj.MatrixForward = M.Box([min, max], inDimensionsOut);
    obj.MatrixBackward = M.Transpose(obj.MatrixForward);
    
    obj.NoiseHidden = RBM.Noise.None;
    obj.NoiseVisible = RBM.Noise.None; // gaussian causes cluster merging

    obj.DeformHidden = RBM.Deform.Sigmoid;
    obj.DeformVisible = RBM.Deform.Sigmoid;

    return obj;
};

RBM.Deform = {
    None:function(inData)
    {
        return inData;
    },
    Sigmoid:function(inData)
    {
        return M.Sigmoid(inData);
    },
    ReLU:function(inData)
    {
        var out = [];
        var value;
        var i, j;
        for(i=0; i<inData.length; i++)
        {
            out.push([]);
            for(j=0; j<inData[i].length; j++)
            {
                value = inData[i][j];
                if(value > 0)
                    out[i][j] = value;
                else
                    out[i][j] = 0;
            }
        }
        return out;
    }
}

RBM.Noise = {
    None:function(inData)
    {
        return inData;
    },
    Bernoulli:function(inData)
    {
        var i, j;
        for(i=0; i<inData.length; i++)
        {
            for(j=0; j<inData[i].length; j++)
            {
                var rand = Math.random();
                if(inData[i][j] > rand)
                    inData[i][j] = 1;
                else
                    inData[i][j] = 0;
            }
        }
        return inData;
    },
    Gaussian:function(inData)
    {
        for(i=0; i<inData.length; i++)
        {
            /* [center coords], radius, pinch, count */
            inData[i] = M.Circle(inData[i], 0.1, 0.01, 1)[0];
        }
        return inData;
    }
};



/*
probability functions. inData must be padded.
*/
//probability of hidden units
RBM.HiddenProbability = function(inRBM, inData)
{
    return  M.Repad( inRBM.DeformHidden(M.Transform(inRBM.MatrixForward, inData)) );
};
//probability of visible units
RBM.VisibleProbability = function(inRBM, inData)
{
    return  M.Repad( inRBM.DeformVisible(M.Transform(inRBM.MatrixBackward, inData)) );
};



RBM.HiddenSample = function(inRBM, inData)
{
    return inRBM.NoiseHidden(RBM.HiddenProbability(inRBM, inData));
};
RBM.VisibleSample = function(inRBM, inData)
{
    return inRBM.NoiseVisible(RBM.VisibleProbability(inRBM, inData));
};



// contrative divergence
RBM.CD = function(inRBM, inData, inCDN, inRate)
{
    var pos;
    var neg;
    var i;

    var visibleProbability;
    var visibleSample;
    var hiddenProbability;
    var hiddenSample;
    var initial;
    var final;

    hiddenSample = RBM.HiddenSample(inRBM, inData);
    initial = hiddenSample;

    for(i=0; i<inCDN; i++)
    {
        visibleSample = RBM.VisibleSample(inRBM, hiddenSample); //v`
        hiddenSample = RBM.HiddenSample(inRBM, visibleSample); //h`
        final = hiddenSample;
    }

    for(i=0; i<inData.length; i++)
    {
        pos = M.Outer(inData[i], initial[i]);
        neg = M.Outer(visibleSample[i], hiddenSample[i]);
        inRBM.MatrixForward = M.Add(inRBM.MatrixForward, M.Scale(pos, inRate));
        inRBM.MatrixForward = M.Subtract(inRBM.MatrixForward, M.Scale(neg, inRate));
    }
    
    inRBM.MatrixBackward = M.Transpose(inRBM.MatrixForward);  
};
//batch of Contrastive Divergence calls
RBM.Train = function(inRBM, inData, inIterations, inCDN, inRate)
{
    var i;
    var copy = M.Pad(M.Clone(inData));
    for(i=0; i<inIterations; i++)
    {
        RBM.CD(inRBM, copy, inCDN, inRate);
    }
};


RBM.Fabricate = function(inRBM, inData, inIterations)
{
    var i;
    var hidden, visible;

    visible = M.Pad(M.Clone(inData));
    for(i=0; i<inIterations; i++)
    {
        hidden = RBM.HiddenProbability(inRBM, visible);
        visible = RBM.VisibleProbability(inRBM, hidden);
    }
    return M.Unpad(visible);
};
RBM.Label = function(inRBM, inData, inIterations)
{
    var i;
    var hidden, visible;

    visible = M.Pad(M.Clone(inData));
    for(i=0; i<inIterations; i++)
    {
        hidden = RBM.HiddenProbability(inRBM, visible);
        visible = RBM.VisibleProbability(inRBM, hidden);
    }
    return M.Unpad(RBM.HiddenProbability(inRBM, visible));
};