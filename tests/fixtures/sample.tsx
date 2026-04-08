import React from 'react';

interface Props {
    name: string;
    count: number;
}

export const MyComponent: React.FC<Props> = ({ name, count }) => {
    const [state, setState] = React.useState(0);
    
    return (
        <div>
            <h1>{name}</h1>
            <p>Count: {count}</p>
        </div>
    );
};

export function HelperFunction() {
    return "helper";
}

export default MyComponent;
